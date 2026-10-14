using System.Globalization;
using System.Text;

namespace SimdYolo;

/// <summary>解析 ONNX 元数据中的 JSON 或 Python 字面量；只读取数据，不执行表达式。</summary>
/// <param name="source">待解析的字典、列表或标量文本。</param>
internal sealed class MetadataLiteral(string source)
{
    /// <summary>下一个待读取字符的位置。</summary>
    private int position;

    /// <summary>读取完整字面量并拒绝尾部表达式。</summary>
    /// <param name="text">JSON 或 Python 字面量文本。</param>
    /// <returns>字典、列表、字符串、整数、布尔值或 null。</returns>
    /// <exception cref="InvalidDataException">文本超限，语法错误，键重复或嵌套过深。</exception>
    internal static object? Parse(string text)
    {
        if (text.Length > 4 * 1024 * 1024) throw new InvalidDataException("Model metadata literal exceeds 4 MiB.");
        var reader = new MetadataLiteral(text); object? value = reader.Value(0); reader.White();
        if (reader.position != text.Length) throw new InvalidDataException("Trailing model metadata expression.");
        return value;
    }

    /// <summary>跳过空白字符。</summary>
    private void White() { while (position < source.Length && char.IsWhiteSpace(source[position])) position++; }

    /// <summary>读取指定标点。</summary>
    /// <param name="character">需要读取的标点字符。</param>
    /// <returns>是否读到该标点；false 不消耗非空白字符。</returns>
    private bool Take(char character) { White(); if (position == source.Length || source[position] != character) return false; position++; return true; }

    /// <summary>读取一个有界嵌套的字面量。</summary>
    /// <param name="depth">当前嵌套深度。</param>
    /// <returns>对应的托管数据。</returns>
    /// <exception cref="InvalidDataException">字面量语法无效或嵌套深度超过 32。</exception>
    private object? Value(int depth)
    {
        White(); if (depth > 32 || position == source.Length) throw new InvalidDataException("Invalid model metadata literal.");
        char current = source[position];
        if (current is '\'' or '"') return String();
        if (Take('{'))
        {
            var map = new Dictionary<object, object?>(); if (Take('}')) return map;
            do
            {
                object key = Value(depth + 1) ?? throw new InvalidDataException("Null metadata key.");
                if (key is not (string or long) || !Take(':')) throw new InvalidDataException("Invalid metadata dictionary key.");
                if (!map.TryAdd(key, Value(depth + 1))) throw new InvalidDataException("Duplicate metadata dictionary key.");
                if (Take('}')) return map;
            } while (Take(',') && !Take('}'));
            if (source[position - 1] == '}') return map;
            throw new InvalidDataException("Invalid metadata dictionary.");
        }
        if (current is '[' or '(')
        {
            position++; char end = current == '[' ? ']' : ')'; var list = new List<object?>(); if (Take(end)) return list;
            do { list.Add(Value(depth + 1)); if (Take(end)) return list; } while (Take(',') && !Take(end));
            if (source[position - 1] == end) return list;
            throw new InvalidDataException("Invalid metadata list.");
        }
        int start = position; while (position < source.Length && !char.IsWhiteSpace(source[position]) && source[position] is not (',' or ':' or '}' or ']' or ')')) position++;
        string token = source[start..position];
        return token switch
        {
            "True" or "true" => true, "False" or "false" => false, "None" or "null" => null,
            _ => long.TryParse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long number) ? (object)number :
                double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double real) && double.IsFinite(real) ? (object)real :
                throw new InvalidDataException("Unsupported model metadata expression.")
        };
    }

    /// <summary>读取带引号字符串，恢复 Python 和 JSON 常用转义及 Unicode 字符。</summary>
    /// <returns>未转义的字符串。</returns>
    /// <exception cref="InvalidDataException">字符串未闭合或转义序列不完整。</exception>
    private string String()
    {
        char quote = source[position++]; var result = new StringBuilder();
        while (position < source.Length)
        {
            char c = source[position++]; if (c == quote) return result.ToString();
            if (c != '\\') { if (c < 32) throw new InvalidDataException("Unescaped metadata control character."); result.Append(c); continue; }
            if (position == source.Length) throw new InvalidDataException("Incomplete metadata escape.");
            c = source[position++];
            if (c is 'x' or 'u' or 'U')
            {
                int length = c == 'x' ? 2 : c == 'u' ? 4 : 8;
                if (position > source.Length - length || !uint.TryParse(source.AsSpan(position, length), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out uint code)) throw new InvalidDataException("Invalid metadata Unicode escape.");
                position += length;
                if (length <= 4) result.Append((char)code);
                else if (code <= 0x10ffff && code is not (>= 0xd800 and <= 0xdfff)) result.Append(char.ConvertFromUtf32((int)code));
                else throw new InvalidDataException("Invalid metadata Unicode scalar.");
            }
            else if (c is '\\' or '\'' or '"' or '/') result.Append(c);
            else if (c is 'n' or 'r' or 't' or 'b' or 'f' or 'a' or 'v') result.Append(c switch { 'n' => '\n', 'r' => '\r', 't' => '\t', 'b' => '\b', 'f' => '\f', 'a' => '\a', _ => '\v' });
            else { result.Append('\\'); result.Append(c); }
        }
        throw new InvalidDataException("Unterminated metadata string.");
    }
}
