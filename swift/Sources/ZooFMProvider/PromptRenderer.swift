import Foundation
import FoundationModels

/// Renders a FoundationModels `Transcript` into the Qwen/Hermes tool-calling
/// ChatML dialect:
///
/// * tools advertised in the system message inside `<tools>…</tools>`, one
///   `{"type":"function","function":{…}}` JSON line each
/// * past tool calls replayed as assistant `<tool_call>{json}</tool_call>`
///   turns, tool results as user-role `<tool_response>…</tool_response>`
/// * reasoning entries are NOT replayed (matches the upstream chat
///   templates, which strip historic thinking)
///
/// LFM2.5 and Granite bundles speak the same `<|im_start|>` ChatML framing,
/// so one renderer covers the current zoo lineup.
enum PromptRenderer {
    /// - Parameter requireToolCall: render a "must call a tool" instruction
    ///   (GenerationOptions.ToolCallingMode.required). Local models have no
    ///   grammar-level enforcement, so this is the honest approximation.
    static func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool = false
    ) -> String {
        var system = "You are a helpful assistant."
        var body = ""

        func append(role: String, _ content: String) {
            body += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                system = segmentsText(instructions.segments)
            case .prompt(let prompt):
                append(role: "user", segmentsText(prompt.segments))
            case .response(let response):
                append(role: "assistant", segmentsText(response.segments))
            case .toolCalls(let calls):
                let rendered = calls.map { call in
                    "<tool_call>\n{\"name\": \"\(call.toolName)\", \"arguments\": \(call.arguments.jsonString)}\n</tool_call>"
                }.joined(separator: "\n")
                append(role: "assistant", rendered)
            case .toolOutput(let output):
                append(
                    role: "user",
                    "<tool_response>\n\(segmentsText(output.segments))\n</tool_response>")
            default:
                continue  // reasoning entries are not replayed into history
            }
        }

        let head =
            "<|im_start|>system\n"
            + systemContent(base: system, tools: tools, requireToolCall: requireToolCall)
            + "<|im_end|>\n"
        return head + body + "<|im_start|>assistant\n"
    }

    /// Appends the `<tools>` block to the system message. No tools — no
    /// block: a plain-chat session's prompt is byte-identical to one rendered
    /// without tool support.
    private static func systemContent(
        base: String,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        guard !tools.isEmpty else { return base }
        var lines: [String] = []
        for tool in tools {
            let schemaJSON: String
            if let data = try? JSONEncoder().encode(tool.parameters),
                let s = String(data: data, encoding: .utf8)
            {
                schemaJSON = s
            } else {
                schemaJSON = "{}"
            }
            lines.append(
                #"{"type": "function", "function": {"name": "\#(tool.name)", "description": "\#(tool.description)", "parameters": \#(schemaJSON)}}"#
            )
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

    private static func segmentsText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }
}
