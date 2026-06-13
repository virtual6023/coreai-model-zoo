import Foundation
import FoundationModels
import Tokenizers

/// A model family's tool-calling "dialect": how tool definitions are
/// advertised in the prompt, how the model marks a tool-call block in its
/// output stream, how that block's body parses into calls, and how past
/// calls/results are replayed into the next prompt.
///
/// Why this exists: in-context format instructions do NOT override a model's
/// training prior. LFM2.5, prompted with the Hermes `<tool_call>` JSON
/// instruction, still emits its native
/// `<|tool_call_start|>[fn(arg="x")]<|tool_call_end|>` pythonic form
/// (verified on-device, gate S6c). Tool calling therefore has to speak each
/// model's native dialect; plain chat is unaffected.
///
/// Known dialects of the current zoo lineup:
/// * qwen3.5 — Hermes: `<tools>` JSON block, `<tool_call>{json}</tool_call>`
///   calls, `<tool_response>` results, ChatML framing → `HermesDialect`.
/// * LFM2.5 — `List of tools: [json…]` in system, pythonic calls between
///   `<|tool_call_start|>`/`<|tool_call_end|>` special tokens, results in
///   `<|tool_response_start|>` wrappers, ChatML framing → `LFMDialect`.
/// * granite-4.0 — Hermes tool syntax verbatim, but `<|start_of_role|>`
///   framing (per its chat template); a future dialect, not yet implemented.
/// * gemma4 — fully custom non-JSON format (`<|tool_call>call:name{…}`,
///   `<|"|>` quote token); recorded, not implemented.
public protocol PromptDialect: Sendable {
    /// Short identifier for logs and debugging.
    var name: String { get }

    /// Marker pair delimiting one tool-call block in the model's output
    /// stream. The stream parser withholds the block's body and hands it to
    /// `parseToolCalls` when the close marker arrives.
    var toolCallOpen: String { get }
    var toolCallClose: String { get }

    /// Render the transcript + tool definitions into the model's prompt
    /// string (the dialect owns the whole render, framing included —
    /// dialects differ in framing, not just tool syntax).
    func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String

    /// Parse the body of ONE complete tool-call block. A block may contain
    /// several calls (LFM emits `[fn1(…), fn2(…)]` in one block). `tools`
    /// provides schema hints for tolerant parsing (e.g. mapping a positional
    /// argument onto a single-parameter tool). Throws
    /// `ZooFMProviderError.malformedToolCall` when nothing in the block is
    /// salvageable.
    func parseToolCalls(
        _ payload: String,
        tools: [Transcript.ToolDefinition]
    ) throws -> [ParsedToolCall]
}

/// One tool call recovered from a model's output block, arguments normalized
/// to a JSON object string (the form `.appendArguments` expects).
public struct ParsedToolCall: Sendable, Equatable {
    public let name: String
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// Picks the dialect for a bundle by probing its tokenizer vocab: models
/// trained with the LFM tool special tokens get the LFM dialect, everything
/// else defaults to Hermes (the de-facto open-model standard).
public func defaultDialect(probing tokenizer: any Tokenizer) -> any PromptDialect {
    if tokenizer.convertTokenToId("<|tool_call_start|>") != nil,
        tokenizer.convertTokenToId("<|tool_call_end|>") != nil
    {
        return LFMDialect()
    }
    return HermesDialect()
}

// MARK: - Shared rendering helpers

extension PromptDialect {
    /// Flattens a transcript entry's segments to prompt text (structured
    /// segments render as their JSON).
    package func segmentsText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }

    /// JSON-encodes a tool's GenerationSchema (`{}` when encoding fails —
    /// an empty schema is still a valid prompt).
    package func schemaJSON(_ tool: Transcript.ToolDefinition) -> String {
        if let data = try? JSONEncoder().encode(tool.parameters),
            let s = String(data: data, encoding: .utf8)
        {
            return s
        }
        return "{}"
    }

    /// Property names of a tool's parameter schema, for positional-argument
    /// mapping. nil when the schema doesn't decode to an object.
    package func parameterNames(_ tool: Transcript.ToolDefinition) -> [String]? {
        guard
            let data = try? JSONEncoder().encode(tool.parameters),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let properties = object["properties"] as? [String: Any]
        else { return nil }
        return Array(properties.keys)
    }
}

/// Escapes a string for embedding inside a JSON string literal.
package func jsonEscaped(_ s: String) -> String {
    var out = ""
    out.reserveCapacity(s.count)
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out
}
