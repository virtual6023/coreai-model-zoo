// zoo-fm-gate — verification harness for ZooFMProvider (macOS).
//
// One scenario per process (a finished turn's background drain keeps the GPU
// busy past the last respond; separate processes keep timings clean):
//
//   swift run -c release zoo-fm-gate <bundle-dir> chat        # plain-chat regress, streamed
//   swift run -c release zoo-fm-gate <bundle-dir> tools       # 2-tool sequential scenario
//   swift run -c release zoo-fm-gate <bundle-dir> multiturn   # 3-turn KV-reuse measurement
//
// ZOO_FM_DEBUG=1 prints the executor's KV fast-path / reset decisions.

import Foundation
import FoundationModels
import ZooFMProvider

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_weather(city: \(arguments.city))")
        ToolCallLog.shared.record(name)
        return "Sunny, 24 degrees Celsius in \(arguments.city)."
    }
}

struct LocalTimeTool: Tool {
    let name = "get_local_time"
    let description = "Get the current local time in a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_local_time(city: \(arguments.city))")
        ToolCallLog.shared.record(name)
        return "It is 8:30 PM in \(arguments.city)."
    }
}

struct FavoriteCityTool: Tool {
    let name = "get_favorite_city"
    let description = "Get the user's favorite city. Takes no input."

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_favorite_city()")
        ToolCallLog.shared.record(name)
        return "The user's favorite city is Sapporo."
    }
}

/// Records which tools the framework actually executed.
final class ToolCallLog: @unchecked Sendable {
    static let shared = ToolCallLog()
    private let lock = NSLock()
    private var calls: [String] = []

    func record(_ name: String) {
        lock.lock()
        calls.append(name)
        lock.unlock()
    }
    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
}

func dumpTranscript(_ transcript: Transcript) {
    print("[transcript]")
    for entry in transcript {
        switch entry {
        case .instructions: print("  - instructions")
        case .prompt: print("  - prompt")
        case .toolCalls(let calls):
            let rendered = calls.map { "\($0.toolName)(\($0.arguments.jsonString))" }
            print("  - toolCalls [\(rendered.joined(separator: ", "))]")
        case .toolOutput(let output):
            print("  - toolOutput \(output.toolName)")
        case .response(let r):
            let text = r.segments.compactMap {
                if case .text(let t) = $0 { return t.content } else { return nil }
            }.joined()
            print("  - response (\(text.count) chars)")
        case .reasoning:
            print("  - reasoning")
        default:
            print("  - other")
        }
    }
}

func dumpUsage(_ usage: LanguageModelSession.Usage, label: String) {
    print(
        "[usage:\(label)] input=\(usage.input.totalTokenCount)"
            + " (cached \(usage.input.cachedTokenCount))"
            + " output=\(usage.output.totalTokenCount)"
            + " (reasoning \(usage.output.reasoningTokenCount))")
}

// MARK: - Mock model (no engine) — verifies FRAMEWORK semantics CPU-only:
// how channel events map to transcript entries, multi-call coalescing, the
// tool round trip, and usage aggregation.

struct MockLanguageModel: LanguageModel {
    typealias Executor = MockExecutor
    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(capabilities: [.toolCalling, .reasoning])
    }
    var executorConfiguration: MockExecutor.Configuration { .init() }
}

struct MockExecutor: LanguageModelExecutor {
    typealias Model = MockLanguageModel
    struct Configuration: Hashable, Sendable {}
    init(configuration: Configuration) throws {}

    nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MockLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Mirrors ZooExecutor's event policy: usage once per turn, attached
        // to the kind of entry the turn produced. (The first version of this
        // mock sent WWDC-339-style upfront usage on `.response` — on the
        // 27.0 beta that materializes an EMPTY Response entry on tool turns,
        // which is exactly why ZooExecutor doesn't do it.)
        let sawToolOutput = request.transcript.contains { entry in
            if case .toolOutput = entry { return true } else { return false }
        }
        if !sawToolOutput {
            // First round: think, then TWO tool calls in one turn.
            await channel.send(.reasoning(action: .appendText("I need weather and time.", tokenCount: 5)))
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: UUID().uuidString, name: "get_weather",
                        action: .appendArguments(#"{"city": "Tokyo"}"#, tokenCount: 4))))
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: UUID().uuidString, name: "get_local_time",
                        action: .appendArguments(#"{"city": "Tokyo"}"#, tokenCount: 4))))
            await channel.send(
                .toolCalls(action: .updateMetadata(["modelID": "mock", "requestID": request.id.uuidString])))
            await channel.send(
                .toolCalls(
                    action: .updateUsage(
                        input: .init(totalTokenCount: 100, cachedTokenCount: 0),
                        output: .init(totalTokenCount: 30, reasoningTokenCount: 5))))
        } else {
            await channel.send(
                .response(action: .appendText("Sunny, 24°C, and it is 8:30 PM in Tokyo.", tokenCount: 12)))
            await channel.send(
                .response(action: .updateMetadata(["modelID": "mock", "requestID": request.id.uuidString])))
            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: 150, cachedTokenCount: 100),
                        output: .init(totalTokenCount: 12, reasoningTokenCount: 0))))
        }
        await Task.yield()
    }
}

@main
struct ZooFMGate {
    static func main() async {
        do {
            try await run()
        } catch {
            FileHandle.standardError.write(Data("FATAL: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        let args = CommandLine.arguments
        if args.count == 2, args[1] == "selftest" {
            try selftest()
            return
        }
        if args.count == 2, args[1] == "mock" {
            try await mock()
            return
        }
        guard args.count >= 3 else {
            print(
                "usage: zoo-fm-gate <bundle-dir> <chat|tools|multiturn>"
                    + " | zoo-fm-gate <selftest|mock>")
            exit(2)
        }
        let bundleURL = URL(fileURLWithPath: args[1])
        let scenario = args[2]

        let t0 = Date()
        let model = try await ZooLanguageModel(resourcesAt: bundleURL)
        print(String(format: "[load] %.2f s", Date().timeIntervalSince(t0)))

        switch scenario {
        case "chat": try await chat(model)
        case "tools": try await tools(model)
        case "toolchain": try await toolchain(model)
        case "multiturn":
            // Optional 4th arg = maximumResponseTokens. Capped turns end by
            // token exhaustion, not EOS — the one case without post-EOS
            // overshoot in the cache, where the append-only fast path can hit.
            let cap = args.count >= 4 ? Int(args[3]) : nil
            try await multiturn(model, maxTokens: cap)
        default:
            print("unknown scenario: \(scenario)")
            exit(2)
        }
    }

    // CPU-only: a real LanguageModelSession over a mock executor (no engine).
    // Verifies the framework-side semantics this provider relies on.
    static func mock() async throws {
        let session = LanguageModelSession(
            model: MockLanguageModel(),
            tools: [WeatherTool(), LocalTimeTool()],
            instructions: "You are a helpful assistant."
        )
        let response = try await session.respond(to: "Weather and time in Tokyo?")
        print("[response] \(response.content)")
        dumpUsage(response.usage, label: "response")
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        var failures: [String] = []
        let executed = ToolCallLog.shared.recorded
        if !executed.contains("get_weather") { failures.append("get_weather not executed") }
        if !executed.contains("get_local_time") { failures.append("get_local_time not executed") }

        var toolCallsEntries = 0
        var callsInFirstEntry = 0
        var emptyResponses = 0
        var reasoningEntries = 0
        for entry in session.transcript {
            switch entry {
            case .toolCalls(let calls):
                toolCallsEntries += 1
                if toolCallsEntries == 1 { callsInFirstEntry = calls.count }
            case .response(let r):
                let text = r.segments.compactMap {
                    if case .text(let t) = $0 { return t.content } else { return nil }
                }.joined()
                if text.isEmpty { emptyResponses += 1 }
            case .reasoning: reasoningEntries += 1
            default: break
            }
        }
        print(
            "[mock] toolCallsEntries=\(toolCallsEntries) callsInFirst=\(callsInFirstEntry)"
                + " emptyResponseEntries=\(emptyResponses) reasoningEntries=\(reasoningEntries)")
        if toolCallsEntries != 1 || callsInFirstEntry != 2 {
            failures.append(
                "expected ONE toolCalls entry with TWO calls, got \(toolCallsEntries)/\(callsInFirstEntry)")
        }
        if emptyResponses > 0 {
            failures.append("upfront metadata/usage materialized \(emptyResponses) empty response entries")
        }
        if response.content.isEmpty { failures.append("final response empty") }

        if failures.isEmpty {
            print("GATE PASS: mock (framework semantics)")
        } else {
            for failure in failures { print("GATE FAIL: \(failure)") }
            exit(1)
        }
    }

    // CPU-only: the streaming tag parser and tool-call JSON parse, fed
    // deliberately awkward chunkings (tags straddling deltas, multi-call
    // turns, partial-marker holds at flush).
    static func selftest() throws {
        func collect(_ chunks: [String]) -> [StreamTagParser.Event] {
            var parser = StreamTagParser()
            var events: [StreamTagParser.Event] = []
            for chunk in chunks { events += parser.consume(chunk) }
            events += parser.flush()
            return events
        }
        func merged(_ events: [StreamTagParser.Event]) -> (
            text: String, reasoning: String, payloads: [String]
        ) {
            var text = "", reasoning = "", payloads: [String] = []
            for event in events {
                switch event {
                case .text(let t): text += t
                case .reasoning(let r): reasoning += r
                case .toolCallPayload(let p): payloads.append(p)
                }
            }
            return (text, reasoning, payloads)
        }
        var failures: [String] = []
        func expect(_ condition: Bool, _ label: String) {
            if !condition { failures.append(label) }
        }

        // 1) think + tool call, tags straddling chunk boundaries
        let r1 = merged(
            collect([
                "<th", "ink>I should call", " the tool.</th", "ink>",
                "<tool", "_call>\n{\"name\": \"get_weather\", \"argum",
                "ents\": {\"city\": \"Tokyo\"}}\n</tool_c", "all>",
            ]))
        expect(r1.text.isEmpty, "1: no plain text expected, got '\(r1.text)'")
        expect(r1.reasoning == "I should call the tool.", "1: reasoning mismatch '\(r1.reasoning)'")
        expect(r1.payloads.count == 1, "1: expected 1 payload, got \(r1.payloads.count)")

        // 2) two tool calls in one turn, text in between
        let r2 = merged(
            collect([
                "Let me check.", "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
                "\n<tool_call>{\"name\":\"b\",\"argum", "ents\":{\"x\":1}}</tool_call>",
            ]))
        expect(r2.payloads.count == 2, "2: expected 2 payloads, got \(r2.payloads.count)")
        expect(r2.text.contains("Let me check."), "2: leading text lost")

        // 3) plain text with a partial-marker hold released at flush
        let r3 = merged(collect(["The answer is 1 <", "2 and 3."]))
        expect(r3.text == "The answer is 1 <2 and 3.", "3: hold-back text lost: '\(r3.text)'")

        // 4) parseToolCall: well-formed, argument-less, malformed
        let c4 = try ZooExecutor.parseToolCall(
            #"{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#)
        expect(c4.name == "get_weather", "4: name mismatch")
        expect(c4.argumentsJSON.contains("Tokyo"), "4: args mismatch")
        let c5 = try ZooExecutor.parseToolCall(#"{"name": "ping"}"#)
        expect(c5.argumentsJSON == "{}", "5: empty args mismatch: \(c5.argumentsJSON)")
        do {
            _ = try ZooExecutor.parseToolCall("not json at all")
            expect(false, "6: malformed payload did not throw")
        } catch is ZooFMProviderError {
            // expected
        }

        // 5) unterminated tool call flushes its partial payload
        let r6 = merged(collect(["<tool_call>{\"name\": \"trunc\""]))
        expect(r6.payloads.count == 1, "7: partial payload not flushed")

        if failures.isEmpty {
            print("GATE PASS: selftest (parser + tool-call JSON)")
        } else {
            for failure in failures { print("GATE FAIL: \(failure)") }
            exit(1)
        }
    }

    // Gate (b): plain chat — no tools attached, so the rendered system prompt
    // carries no tools block. Streams via streamResponse to prove textDelta
    // immediacy (first-delta latency ≪ turn time).
    static func chat(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model, instructions: "You are a helpful assistant.")

        // Unlike Apple's adapter (wrong prewarm signature → silent no-op),
        // ZooExecutor's prewarm actually runs — this should take visible time
        // and make turn 1's first delta faster.
        let tw = Date()
        session.prewarm()
        print(String(format: "[prewarm] %.2f s", Date().timeIntervalSince(tw)))

        for prompt in ["Why is the sky blue? Answer in two sentences.", "And at sunset?"] {
            print("[prompt] \(prompt)")
            let t = Date()
            var firstDelta: TimeInterval?
            var previous = ""
            let stream = session.streamResponse(to: prompt)
            for try await snapshot in stream {
                let content = snapshot.content
                if firstDelta == nil, !content.isEmpty {
                    firstDelta = Date().timeIntervalSince(t)
                }
                let delta = String(content.dropFirst(previous.count))
                FileHandle.standardOutput.write(Data(delta.utf8))
                previous = content
            }
            print()
            let dt = Date().timeIntervalSince(t)
            print(
                String(
                    format: "[turn] %.2f s (first delta %.2f s, %d chars)",
                    dt, firstDelta ?? -1, previous.count))
            guard !previous.isEmpty else {
                print("GATE FAIL: empty response")
                exit(1)
            }
        }
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)
        print("GATE PASS: chat")
    }

    // Gate (a): two tools, a prompt needing both. The model may call them
    // one per respond round (weather → toolOutput → time → toolOutput →
    // answer) or both in one turn; the gate requires that BOTH executed and
    // the final answer is grounded on both results.
    static func tools(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [WeatherTool(), LocalTimeTool()],
            instructions: "You are a helpful assistant."
        )

        let prompt = "What is the weather in Tokyo right now, and what time is it there?"
        print("[prompt] \(prompt)")
        let t = Date()
        let response = try await session.respond(to: prompt)
        let dt = Date().timeIntervalSince(t)
        print("[response] \(response.content)")
        print(String(format: "[turn] %.2f s", dt))
        dumpUsage(response.usage, label: "response")
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        let executed = Set(ToolCallLog.shared.recorded)
        let answer = response.content.lowercased()
        var failures: [String] = []
        if !executed.contains("get_weather") { failures.append("get_weather never executed") }
        if !executed.contains("get_local_time") { failures.append("get_local_time never executed") }
        if !answer.contains("24") { failures.append("answer not grounded on weather result") }
        if !(answer.contains("8:30") || answer.contains("20:30")) {
            failures.append("answer not grounded on time result")
        }
        if failures.isEmpty {
            print("GATE PASS: tools (executed: \(ToolCallLog.shared.recorded.joined(separator: " → ")))")
        } else {
            print("GATE FAIL: \(failures.joined(separator: "; "))")
            exit(1)
        }
    }

    // Sequential tool chain: the second call DEPENDS on the first's result
    // (city unknown until get_favorite_city returns), so the model must call
    // a tool on the respond AFTER a toolOutput — the cross-round path.
    static func toolchain(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [FavoriteCityTool(), WeatherTool()],
            instructions: "You are a helpful assistant."
        )
        let prompt = "What is the weather in my favorite city right now?"
        print("[prompt] \(prompt)")
        let t = Date()
        let response = try await session.respond(to: prompt)
        let dt = Date().timeIntervalSince(t)
        print("[response] \(response.content)")
        print(String(format: "[turn] %.2f s", dt))
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        let executed = ToolCallLog.shared.recorded
        let answer = response.content.lowercased()
        var failures: [String] = []
        if executed.first != "get_favorite_city" {
            failures.append("expected get_favorite_city first, got \(executed)")
        }
        if !executed.contains("get_weather") { failures.append("get_weather never executed") }
        if !answer.contains("sapporo") { failures.append("answer not grounded on favorite city") }
        if !answer.contains("24") { failures.append("answer not grounded on weather result") }
        if failures.isEmpty {
            print("GATE PASS: toolchain (executed: \(executed.joined(separator: " → ")))")
        } else {
            print("GATE FAIL: \(failures.joined(separator: "; "))")
            exit(1)
        }
    }

    // Stretch harness: 3 short turns; with ZOO_FM_DEBUG=1 the executor logs
    // whether each turn reused the cache (fast path) or reset. Reports
    // per-turn latency and cachedTokenCount so the multi-turn tax is visible.
    static func multiturn(_ model: ZooLanguageModel, maxTokens: Int?) async throws {
        let session = LanguageModelSession(
            model: model,
            instructions: maxTokens != nil
                ? "You are a helpful assistant. Answer thoroughly and at length."
                : "You are a terse assistant. Answer in one short sentence.")
        let options =
            maxTokens.map { GenerationOptions(maximumResponseTokens: $0) } ?? GenerationOptions()
        if let maxTokens {
            print("[multiturn] maximumResponseTokens=\(maxTokens) (turns end by cap, not EOS)")
        }
        // With a cap, prompts must want MORE tokens than the cap so turns end
        // by exhaustion (no post-EOS overshoot in the cache → the append-only
        // fast path can hit). Without one, terse prompts measure the
        // EOS-ended (overshoot) regime.
        let prompts =
            maxTokens != nil
            ? [
                "Describe the planet Mars in detail.",
                "Now describe Jupiter the same way.",
                "Compare the two planets you described.",
            ]
            : [
                "Name one planet.",
                "Name another one.",
                "Which of those two is bigger?",
            ]
        for (index, prompt) in prompts.enumerated() {
            print("[prompt \(index + 1)] \(prompt)")
            let t = Date()
            let response = try await session.respond(to: prompt, options: options)
            let dt = Date().timeIntervalSince(t)
            print("[response] \(response.content)")
            print(String(format: "[turn %d] %.2f s", index + 1, dt))
            dumpUsage(response.usage, label: "turn\(index + 1)")
        }
        dumpTranscript(session.transcript)
        print("GATE PASS: multiturn (latencies above; see ZOO_FM_DEBUG for reuse decisions)")
    }
}
