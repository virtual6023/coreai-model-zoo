import Foundation
import FoundationModels

/// LFM2.5's native tool-calling dialect, per the `chat_template.jinja`
/// shipped in the zoo bundles (primary source) and the on-device emission
/// observed in gate S6c:
///
/// * tools advertised as plain text in the system message:
///   `List of tools: [{"name": …, "description": …, "parameters": {…}}, …]`
///   (appended to the user instructions with a newline, exactly like the
///   template does)
/// * calls emitted between the `<|tool_call_start|>`/`<|tool_call_end|>`
///   special tokens in PYTHONIC form: `[get_weather(city="Tokyo"),
///   get_local_time(city="Tokyo")]` — one block, possibly several calls
/// * tool results replayed as a `tool`-role ChatML turn wrapping the body in
///   `<|tool_response_start|>`/`<|tool_response_end|>`
/// * framing is ChatML (`<|im_start|>role…<|im_end|>`), BOS is added by the
///   tokenizer (`add_bos_token: true`)
/// * the template renders NO system block when there are neither
///   instructions nor tools — matched here (unlike Hermes, which injects a
///   default system prompt; native fidelity wins for native dialects)
///
/// The pythonic argument parser is deliberately tolerant — S6c showed the
/// model emits half-mangled argument lists (bare values, single quotes,
/// Python literals). See `PythonicCallParser`.
public struct LFMDialect: PromptDialect {
    public let name = "lfm"
    public let toolCallOpen = "<|tool_call_start|>"
    public let toolCallClose = "<|tool_call_end|>"

    public init() {}

    public func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool = false
    ) -> String {
        renderChatML(
            transcript: transcript,
            defaultSystem: "",
            head: { system in
                // The template renders NO system block when there is neither
                // instructions nor tools (native fidelity, unlike Hermes' default).
                let systemText = self.systemContent(
                    base: system, tools: tools, requireToolCall: requireToolCall)
                return systemText.isEmpty ? "" : "<|im_start|>system\n\(systemText)<|im_end|>\n"
            },
            toolEntry: { entry in
                switch entry {
                case .toolCalls(let calls):
                    // Pythonic call syntax, not JSON — names are bare identifiers,
                    // so no JSON escaping applies here (string VALUES are escaped
                    // inside pythonicCall).
                    let rendered = calls.map { call in
                        self.pythonicCall(
                            name: call.toolName, argumentsJSON: call.arguments.jsonString)
                    }.joined(separator: ", ")
                    return ("assistant", "<|tool_call_start|>[\(rendered)]<|tool_call_end|>")
                case .toolOutput(let output):
                    return (
                        "tool",
                        "<|tool_response_start|>\(self.segmentsText(output.segments))<|tool_response_end|>"
                    )
                default:
                    return nil  // reasoning entries are not replayed into history
                }
            })
    }

    /// `system + "\n" + "List of tools: [json, json]"` — the template's
    /// exact concatenation (newline only when both parts are present).
    private func systemContent(
        base: String,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        guard !tools.isEmpty else { return base }
        let toolJSONs = tools.map { toolDescriptorJSON($0) }
        var out = base
        if !out.isEmpty { out += "\n" }
        out += "List of tools: [\(toolJSONs.joined(separator: ", "))]"
        if requireToolCall {
            out += "\nYou MUST call a tool to answer; do not answer directly."
        }
        return out
    }

    /// Replays a transcript tool call in the form the model itself emits:
    /// `name(key="value", n=3)`. Keys are sorted for deterministic renders
    /// (KV-prefix stability); values use Python literals (`True`/`None`)
    /// since that is the emission format the model was trained on.
    func pythonicCall(name: String, argumentsJSON: String) -> String {
        guard
            let data = argumentsJSON.data(using: .utf8),
            let object = (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]))
                as? [String: Any]
        else {
            // Non-object arguments (fragment) — replay raw inside the parens.
            return "\(name)(\(argumentsJSON))"
        }
        let kwargs = object.keys.sorted().map { key in
            "\(key)=\(pythonLiteral(object[key]!))"
        }
        return "\(name)(\(kwargs.joined(separator: ", ")))"
    }

    private func pythonLiteral(_ value: Any) -> String {
        switch value {
        case let s as String:
            return "\"\(jsonEscaped(s))\""
        case let n as NSNumber:
            if n.isBool { return n.boolValue ? "True" : "False" }
            return "\(n)"
        case is NSNull:
            return "None"
        case let array as [Any]:
            return "[\(array.map(pythonLiteral).joined(separator: ", "))]"
        case let dict as [String: Any]:
            let items = dict.keys.sorted().map { key in
                "\"\(jsonEscaped(key))\": \(pythonLiteral(dict[key]!))"
            }
            return "{\(items.joined(separator: ", "))}"
        default:
            return "\(value)"
        }
    }

    /// Parses `[fn1(a=1, b="x"), fn2()]` (outer brackets optional). Calls
    /// that fail to parse are skipped (the rest of the block still
    /// executes); throws only when the block yields nothing.
    public func parseToolCalls(
        _ payload: String,
        tools: [Transcript.ToolDefinition]
    ) throws -> [ParsedToolCall] {
        var hints: [String: [String]] = [:]
        for tool in tools {
            if let names = parameterNames(tool) { hints[tool.name] = names }
        }
        var parser = PythonicCallParser(payload, parameterHints: hints)
        let calls = parser.parseCalls()
        guard !calls.isEmpty else {
            throw ZooFMProviderError.malformedToolCall(
                payload: payload.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return calls
    }
}

extension NSNumber {
    /// JSONSerialization booleans are CFBoolean-backed NSNumbers; this is the
    /// reliable discriminator (objCType "c" is shared with Int8).
    var isBool: Bool {
        CFGetTypeID(self) == CFBooleanGetTypeID()
    }
}

// MARK: - Pythonic call parser

/// Tolerant scanner for LFM's pythonic call lists. The model's argument
/// lists arrive half-mangled (S6c), so beyond the canonical
/// `[fn(key="value")]` it accepts: missing outer brackets, single-quoted or
/// bare/unquoted values (scanned to the next delimiter, so spaces survive),
/// Python literals (`True`/`False`/`None`) and their JSON spellings, nested
/// lists/dicts, `:` for `=`, quoted keyword names, trailing commas, and
/// truncated tails (unterminated strings/parens take what is there). One
/// positional argument maps onto the tool's parameter when the schema has
/// exactly one; otherwise the call is dropped. A call that fails to parse is
/// skipped by scanning to the next top-level comma.
package struct PythonicCallParser {
    private let chars: [Character]
    private var i = 0
    private let parameterHints: [String: [String]]

    package init(_ s: String, parameterHints: [String: [String]] = [:]) {
        self.chars = Array(s)
        self.parameterHints = parameterHints
    }

    package mutating func parseCalls() -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        skipWhitespace()
        if peek == "[" { advance() }
        while true {
            skipWhitespace()
            while peek == "," { advance(); skipWhitespace() }
            guard let c = peek, c != "]" else { break }
            if let call = parseCall() {
                calls.append(call)
            } else {
                skipToNextTopLevelComma()
            }
        }
        return calls
    }

    private mutating func parseCall() -> ParsedToolCall? {
        guard let name = parseIdentifier(), !name.isEmpty else { return nil }
        skipWhitespace()
        guard peek == "(" else { return nil }
        advance()

        var arguments: [String: Any] = [:]
        var positionals: [Any] = []
        while true {
            skipWhitespace()
            guard let c = peek else { break }  // truncated tail — take what we have
            if c == ")" { advance(); break }
            // A stray/mismatched closer (`]` or `}`) ends the arg list. None of
            // the value parsers consume a closer, so without this the loop would
            // spin forever on the frozen position (the hang this tolerant parser
            // shipped with — reachable from realistic mangled output like
            // `get_weather(city="Tokyo"})`). Treating it as a terminator both
            // guarantees progress and salvages the arguments parsed so far.
            // parseBareword is deliberately NOT changed to consume closers —
            // `fn(a=)` needs its `)` to survive as the terminator.
            if c == "]" || c == "}" { break }
            if c == "," { advance(); continue }

            let saved = i
            if let key = parseKeywordName() {
                arguments[key] = parseValue()
            } else {
                i = saved
                positionals.append(parseValue())
            }
        }

        if !positionals.isEmpty {
            // Schema hint: a single-parameter tool takes the lone positional.
            guard
                positionals.count == 1,
                let names = parameterHints[name], names.count == 1,
                arguments.isEmpty
            else { return nil }
            arguments[names[0]] = positionals[0]
        }

        guard
            let data = try? JSONSerialization.data(
                withJSONObject: arguments, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return nil }
        return ParsedToolCall(name: name, argumentsJSON: json)
    }

    /// `ident =` or `"ident" :` — returns nil (position untouched by caller
    /// restore) when what follows is not a keyword assignment.
    private mutating func parseKeywordName() -> String? {
        skipWhitespace()
        let key: String?
        if peek == "\"" || peek == "'" {
            key = parseQuotedString()
        } else {
            key = parseIdentifier()
        }
        guard let key, !key.isEmpty else { return nil }
        skipWhitespace()
        guard peek == "=" || peek == ":" else { return nil }
        advance()
        return key
    }

    private mutating func parseValue() -> Any {
        skipWhitespace()
        switch peek {
        case "\"", "'":
            return parseQuotedString() ?? ""
        case "[":
            advance()
            var items: [Any] = []
            while true {
                skipWhitespace()
                guard let c = peek else { break }
                if c == "]" { advance(); break }
                // Foreign closer (`)`/`}`) ends the list — same no-spin
                // terminator rule as the argument loop.
                if c == ")" || c == "}" { break }
                if c == "," { advance(); continue }
                items.append(parseValue())
            }
            return items
        case "{":
            advance()
            var dict: [String: Any] = [:]
            while true {
                skipWhitespace()
                guard let c = peek else { break }
                if c == "}" { advance(); break }
                // Foreign closer (`)`/`]`) ends the dict — same no-spin
                // terminator rule as the argument loop.
                if c == ")" || c == "]" { break }
                if c == "," { advance(); continue }
                let key: String?
                if peek == "\"" || peek == "'" {
                    key = parseQuotedString()
                } else {
                    key = parseIdentifier()
                }
                skipWhitespace()
                if peek == ":" || peek == "=" { advance() }
                let value = parseValue()
                if let key, !key.isEmpty { dict[key] = value }
            }
            return dict
        default:
            return parseBareword()
        }
    }

    /// Unquoted value: scan to the next structural delimiter, trim, then
    /// classify (number / boolean / null) with plain-string fallback —
    /// `city=Tokyo` and `city=New York` both come out as strings.
    private mutating func parseBareword() -> Any {
        var word = ""
        while let c = peek, c != "," && c != ")" && c != "]" && c != "}" {
            word.append(c)
            advance()
        }
        let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case "True", "true": return true
        case "False", "false": return false
        case "None", "null", "nil": return NSNull()
        default: break
        }
        if let intValue = Int(trimmed) { return intValue }
        if let doubleValue = Double(trimmed) { return doubleValue }
        return trimmed
    }

    /// Backslash escapes: `\n`/`\t`/`\r` decode to the control character; any
    /// other escape yields the literal following character (so `\"`, `\\`, `\'`
    /// work, but `\b`/`\f` become "b"/"f" and `\uXXXX` is NOT decoded — the "u"
    /// and hex digits survive literally). Full JSON unescaping isn't needed for
    /// the pythonic argument values the model emits. An unterminated string
    /// (stream truncation) takes the rest of the input.
    private mutating func parseQuotedString() -> String? {
        guard let quote = peek, quote == "\"" || quote == "'" else { return nil }
        advance()
        var out = ""
        while let c = peek {
            advance()
            if c == "\\" {
                if let escaped = peek {
                    advance()
                    switch escaped {
                    case "n": out.append("\n")
                    case "t": out.append("\t")
                    case "r": out.append("\r")
                    default: out.append(escaped)
                    }
                }
                continue
            }
            if c == quote { return out }
            out.append(c)
        }
        return out  // unterminated — truncated stream tail
    }

    private mutating func parseIdentifier() -> String? {
        skipWhitespace()
        var out = ""
        while let c = peek, c.isLetter || c.isNumber || c == "_" || c == "." || c == "-" {
            out.append(c)
            advance()
        }
        return out.isEmpty ? nil : out
    }

    /// Error recovery: skip ahead to the comma separating this call from the
    /// next, honoring nesting and quotes so commas inside argument values
    /// don't end the skip early.
    private mutating func skipToNextTopLevelComma() {
        var depth = 0
        while let c = peek {
            advance()
            switch c {
            case "(", "[", "{":
                depth += 1
            case ")", "]", "}":
                depth -= 1
                if depth < 0 { return }  // closed the outer list
            case "\"", "'":
                while let inner = peek {
                    advance()
                    if inner == "\\" { if peek != nil { advance() }; continue }
                    if inner == c { break }
                }
            case ",":
                if depth == 0 { return }
            default:
                break
            }
        }
    }

    private var peek: Character? {
        i < chars.count ? chars[i] : nil
    }

    private mutating func advance() {
        i += 1
    }

    private mutating func skipWhitespace() {
        while let c = peek, c.isWhitespace { advance() }
    }
}
