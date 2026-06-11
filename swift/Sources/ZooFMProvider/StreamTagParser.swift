import Foundation

/// Streaming segmenter for a tool-calling model's text deltas.
///
/// Extends the marker-pair pattern of Apple's `ThinkTagParser` (in
/// `CoreAILanguageModels`) to two tag families:
///
/// * `<think>…</think>` (marker pair configurable) — body streams out as
///   `.reasoning` events, delta by delta.
/// * `<tool_call>…</tool_call>` — body is NOT streamed; it accumulates and is
///   emitted as one `.toolCallPayload` event when the closing tag arrives, so
///   the caller hands the framework a complete JSON object per call.
///
/// Everything else streams out as `.text` the moment it can be proven not to
/// be the start of a marker: the parser holds back at most `marker.count - 1`
/// trailing characters per candidate marker, so tags that straddle token
/// boundaries are still caught (same hold-back contract as `ThinkTagParser`).
///
/// Feed incremental detokenizer output via `consume(_:)`; call `flush()` once
/// at end of stream — an unterminated `<tool_call>` body is flushed as a
/// (likely truncated) `.toolCallPayload` and left to the JSON parser to
/// accept or reject.
package struct StreamTagParser {
    package enum Event: Equatable {
        case text(String)
        case reasoning(String)
        case toolCallPayload(String)
    }

    private enum Mode {
        case text
        case reasoning
        case toolCall
    }

    private let thinkOpen: String
    private let thinkClose: String
    private let toolOpen = "<tool_call>"
    private let toolClose = "</tool_call>"

    private var buffer = ""
    private var mode: Mode = .text
    private var toolPayload = ""

    package init(thinkOpen: String = "<think>", thinkClose: String = "</think>") {
        self.thinkOpen = thinkOpen
        self.thinkClose = thinkClose
    }

    package mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        return drain(isFinal: false)
    }

    /// Emit pending content as final events. Unclosed reasoning flushes as
    /// `.reasoning`; an unclosed tool call flushes its partial payload.
    package mutating func flush() -> [Event] {
        drain(isFinal: true)
    }

    private mutating func drain(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            switch mode {
            case .text:
                // First marker in the buffer wins (a model emits either a
                // thinking block or a tool call next, never both at once).
                let thinkRange = buffer.range(of: thinkOpen)
                let toolRange = buffer.range(of: toolOpen)
                let hit: (Range<String.Index>, Mode)?
                switch (thinkRange, toolRange) {
                case (let t?, let c?):
                    hit = t.lowerBound < c.lowerBound ? (t, .reasoning) : (c, .toolCall)
                case (let t?, nil): hit = (t, .reasoning)
                case (nil, let c?): hit = (c, .toolCall)
                case (nil, nil): hit = nil
                }
                if let (range, next) = hit {
                    let before = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !before.isEmpty { events.append(.text(before)) }
                    buffer = String(buffer[range.upperBound...])
                    mode = next
                    continue
                }
                let safe = isFinal
                    ? buffer.endIndex
                    : lastSafeIndex(forTags: [thinkOpen, toolOpen])
                if safe > buffer.startIndex {
                    events.append(.text(String(buffer[buffer.startIndex..<safe])))
                    buffer = String(buffer[safe...])
                }
                return events

            case .reasoning:
                if let range = buffer.range(of: thinkClose) {
                    let body = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !body.isEmpty { events.append(.reasoning(body)) }
                    buffer = String(buffer[range.upperBound...])
                    mode = .text
                    continue
                }
                let safe = isFinal ? buffer.endIndex : lastSafeIndex(forTags: [thinkClose])
                if safe > buffer.startIndex {
                    events.append(.reasoning(String(buffer[buffer.startIndex..<safe])))
                    buffer = String(buffer[safe...])
                }
                return events

            case .toolCall:
                if let range = buffer.range(of: toolClose) {
                    toolPayload += String(buffer[buffer.startIndex..<range.lowerBound])
                    buffer = String(buffer[range.upperBound...])
                    events.append(.toolCallPayload(toolPayload))
                    toolPayload = ""
                    mode = .text
                    continue
                }
                // Accumulate silently; keep a possible partial close tag in
                // the buffer.
                let safe = isFinal ? buffer.endIndex : lastSafeIndex(forTags: [toolClose])
                if safe > buffer.startIndex {
                    toolPayload += String(buffer[buffer.startIndex..<safe])
                    buffer = String(buffer[safe...])
                }
                if isFinal, !toolPayload.isEmpty {
                    events.append(.toolCallPayload(toolPayload))
                    toolPayload = ""
                }
                return events
            }
        }
    }

    /// Rightmost index such that the suffix from there to end-of-buffer is
    /// not a non-empty prefix of any candidate tag. Scans at most
    /// `max(tag.count) - 1` trailing characters.
    private func lastSafeIndex(forTags tags: [String]) -> String.Index {
        let maxHold = (tags.map(\.count).max() ?? 1) - 1
        guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
        let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
        var idx = holdStart
        while idx < buffer.endIndex {
            let suffix = buffer[idx...]
            if tags.contains(where: { $0.starts(with: suffix) }) {
                return idx
            }
            idx = buffer.index(after: idx)
        }
        return buffer.endIndex
    }
}
