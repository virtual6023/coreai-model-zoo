import Foundation
import FoundationModels

/// The Qwen/Hermes tool-calling ChatML dialect (the de-facto open-model
/// standard, and the zoo default):
///
/// * tools advertised in the system message inside `<tools>…</tools>`, one
///   `{"type":"function","function":{…}}` JSON line each
/// * calls emitted as `<tool_call>{json}</tool_call>` blocks, one JSON
///   object per block
/// * past tool calls replayed as assistant `<tool_call>{json}</tool_call>`
///   turns, tool results as user-role `<tool_response>…</tool_response>`
/// * reasoning entries are NOT replayed (matches the upstream chat
///   templates, which strip historic thinking)
///
/// Verified on qwen3.5 (gates S3–S5). granite-4.0 uses the same tool syntax
/// but `<|start_of_role|>` framing — it needs its own dialect, not this one.
public struct HermesDialect: PromptDialect {
    public let name = "hermes"
    public let toolCallOpen = "<tool_call>"
    public let toolCallClose = "</tool_call>"

    public init() {}

    /// - Parameter requireToolCall: render a "must call a tool" instruction
    ///   (GenerationOptions.ToolCallingMode.required). Local models have no
    ///   grammar-level enforcement, so this is the honest approximation.
    public func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool = false
    ) -> String {
        renderChatML(
            transcript: transcript,
            defaultSystem: "You are a helpful assistant.",
            head: { system in
                "<|im_start|>system\n"
                    + self.systemContent(
                        base: system, tools: tools, requireToolCall: requireToolCall)
                    + "<|im_end|>\n"
            },
            toolEntry: { entry in
                switch entry {
                case .toolCalls(let calls):
                    // toolName is JSON-escaped (it is interpolated into a JSON
                    // string); arguments.jsonString is already valid JSON.
                    let rendered = calls.map { call in
                        "<tool_call>\n{\"name\": \"\(jsonEscaped(call.toolName))\", \"arguments\": \(call.arguments.jsonString)}\n</tool_call>"
                    }.joined(separator: "\n")
                    return ("assistant", rendered)
                case .toolOutput(let output):
                    return (
                        "user",
                        "<tool_response>\n\(self.segmentsText(output.segments))\n</tool_response>")
                default:
                    return nil  // reasoning entries are not replayed into history
                }
            })
    }

    /// Appends the `<tools>` block to the system message. No tools — no
    /// block: a plain-chat session's prompt is byte-identical to one rendered
    /// without tool support.
    private func systemContent(
        base: String,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        guard !tools.isEmpty else { return base }
        var lines: [String] = []
        for tool in tools {
            // Shared, JSON-escaped descriptor wrapped in Hermes' function envelope.
            lines.append(#"{"type": "function", "function": \#(toolDescriptorJSON(tool))}"#)
        }
        let requirement = requireToolCall
            ? "\nYou MUST respond with a function call; do not answer directly.\n"
            : ""
        return base + """


            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            \(lines.joined(separator: "\n"))
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call>
            \(requirement)
            """
    }

    /// `{"name": "...", "arguments": {...}}` per block; a top-level JSON
    /// ARRAY of such objects is tolerated (some Hermes-tuned models emit
    /// multi-call arrays). Anything else throws.
    public func parseToolCalls(
        _ payload: String,
        tools: [Transcript.ToolDefinition]
    ) throws -> [ParsedToolCall] {
        let trimmed = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let data = trimmed.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: data)
        else {
            throw ZooFMProviderError.malformedToolCall(payload: trimmed)
        }
        let objects: [[String: Any]]
        if let object = root as? [String: Any] {
            objects = [object]
        } else if let array = root as? [[String: Any]], !array.isEmpty {
            objects = array
        } else {
            throw ZooFMProviderError.malformedToolCall(payload: trimmed)
        }
        return try objects.map { object in
            guard let name = object["name"] as? String, !name.isEmpty else {
                throw ZooFMProviderError.malformedToolCall(payload: trimmed)
            }
            let arguments = object["arguments"] ?? [String: Any]()
            guard
                let argumentsData = try? JSONSerialization.data(
                    withJSONObject: arguments, options: [.fragmentsAllowed, .sortedKeys]),
                let argumentsJSON = String(data: argumentsData, encoding: .utf8)
            else {
                throw ZooFMProviderError.malformedToolCall(payload: trimmed)
            }
            return ParsedToolCall(name: name, argumentsJSON: argumentsJSON)
        }
    }
}
