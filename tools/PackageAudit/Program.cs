using System.Reflection;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;

/// <summary>通过标准库读取主程序集和编译 XML 文档，不创建模型或计算后端。</summary>
internal static class Program
{
    /// <summary>审核 PE 托管标志、引用和 XML 与实际声明的参数、返回值及属性文档。</summary>
    /// <param name="args">主 DLL、XML 文档和 JSON 输出路径。</param>
    /// <returns>已完成结构审核为 0，读取或解析失败为 1；具体文档缺项写入 JSON。</returns>
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length != 3) throw new ArgumentException("PackageAudit DLL XML OUTPUT.json");
            string dll = Path.GetFullPath(args[0]);
            // 训练程序集依赖同目录主库；只从被审查输出目录解析，仍记录实际程序集引用。
            AssemblyLoadContext.Default.Resolving += (_, name) =>
            {
                string candidate = Path.Combine(Path.GetDirectoryName(dll)!, name.Name + ".dll");
                return File.Exists(candidate) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(candidate) : null;
            };
            using var stream = File.OpenRead(dll); using var pe = new PEReader(stream);
            bool ilOnly = pe.PEHeaders.CorHeader is { } cor && (cor.Flags & CorFlags.ILOnly) != 0;
            var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(dll);
            string[] references = assembly.GetReferencedAssemblies().Select(a => a.Name ?? "").Order().ToArray();
            var document = XDocument.Load(args[1]);
            var members = document.Root!.Element("members")!.Elements("member").ToDictionary(e => (string)e.Attribute("name")!);
            var mapped = new HashSet<string>(); var problems = new List<object>(); var undocumentedPublic = new List<string>();
            var generatedPropertySummaries = new List<string>(); var parameterlessConstructors = new List<string>();
            int methodChecks = 0, parameterChecks = 0, returnChecks = 0, voidMethods = 0, constructorChecks = 0, propertyChecks = 0;
            const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            foreach (var pair in members)
            {
                if (!Chinese(pair.Value.Element("summary")?.Value)) problems.Add(new { Member = pair.Key, Issue = "MissingChineseSummary" });
                foreach (var child in pair.Value.Elements().Where(e => e.Name.LocalName is "param" or "typeparam" or "returns" or "value" or "exception"))
                    if (!Chinese(child.Value)) problems.Add(new { Member = pair.Key, Issue = "NonChineseOrEmptyTag", Tag = child.Name.LocalName, Name = (string?)child.Attribute("name") });
            }
            foreach (Type type in assembly.GetTypes())
            {
                string typeName = type.FullName!.Replace('+', '.'); string typeId = "T:" + typeName;
                if (members.ContainsKey(typeId)) mapped.Add(typeId);
                else if (type.IsVisible && !Generated(type)) undocumentedPublic.Add(typeId);
                foreach (var method in type.GetMethods(flags).Cast<MethodBase>().Concat(type.GetConstructors(flags)))
                {
                    string id = MethodId(method);
                    bool accessor = method.IsSpecialName && (method.Name.StartsWith("get_", StringComparison.Ordinal) || method.Name.StartsWith("set_", StringComparison.Ordinal) || method.Name.StartsWith("add_", StringComparison.Ordinal) || method.Name.StartsWith("remove_", StringComparison.Ordinal));
                    if (!members.TryGetValue(id, out var entry))
                    {
                        if (type.IsVisible && method.IsPublic && !accessor && !Generated(method) && !Generated(type))
                        {
                            if (method is ConstructorInfo && method.GetParameters().Length == 0 && members.ContainsKey(typeId)) parameterlessConstructors.Add(id);
                            else undocumentedPublic.Add(id);
                        }
                        continue;
                    }
                    mapped.Add(id); methodChecks++;
                    var parameterNames = method.GetParameters().Select(p => p.Name!).ToArray();
                    foreach (string name in parameterNames)
                    {
                        parameterChecks++;
                        if (!entry.Elements("param").Any(e => (string?)e.Attribute("name") == name)) problems.Add(new { Member = id, Issue = "MissingParameter", Name = name });
                    }
                    foreach (var parameter in entry.Elements("param"))
                        if (!parameterNames.Contains((string?)parameter.Attribute("name"))) problems.Add(new { Member = id, Issue = "UnknownParameter", Name = (string?)parameter.Attribute("name") });
                    if (method is ConstructorInfo) constructorChecks++;
                    else if (((MethodInfo)method).ReturnType == typeof(void)) voidMethods++;
                    else
                    {
                        returnChecks++;
                        if (entry.Element("returns") is null) problems.Add(new { Member = id, Issue = "MissingReturns" });
                    }
                }
                foreach (var property in type.GetProperties(flags))
                {
                    string id = "P:" + typeName + "." + property.Name.Replace('.', '#');
                    var indices = property.GetIndexParameters();
                    if (indices.Length > 0) id += "(" + string.Join(',', indices.Select(p => TypeId(p.ParameterType))) + ")";
                    if (members.TryGetValue(id, out var entry))
                    {
                        mapped.Add(id); propertyChecks++;
                        if (entry.Element("value") is null)
                        {
                            if (Generated(property.GetMethod) || Generated(property.SetMethod)) generatedPropertySummaries.Add(id);
                            else problems.Add(new { Member = id, Issue = "MissingPropertyValue" });
                        }
                    }
                    else if (type.IsVisible && (property.GetMethod?.IsPublic == true || property.SetMethod?.IsPublic == true) && !Generated(property) && !Generated(type) && !Generated(property.GetMethod ?? (MemberInfo?)property.SetMethod)) undocumentedPublic.Add(id);
                }
                foreach (var field in type.GetFields(flags))
                {
                    string id = "F:" + typeName + "." + field.Name;
                    if (members.ContainsKey(id)) mapped.Add(id);
                    else if (type.IsVisible && field.IsPublic && !field.IsSpecialName && !Generated(field) && !Generated(type)) undocumentedPublic.Add(id);
                }
                foreach (var eventInfo in type.GetEvents(flags))
                {
                    string id = "E:" + typeName + "." + eventInfo.Name;
                    if (members.ContainsKey(id)) mapped.Add(id);
                }
            }
            var report = new
            {
                Version = assembly.GetName().Version!.ToString(3), IlOnly = ilOnly, AssemblyReferences = references,
                ExternalAssemblyReferences = references.Where(name => name is not ("netstandard" or "mscorlib" or "Microsoft.CSharp") && !name.StartsWith("System", StringComparison.Ordinal)).ToArray(),
                XmlMembers = members.Count, ChineseXmlSummaries = members.Count(pair => Chinese(pair.Value.Element("summary")?.Value)),
                MappedXmlMembers = mapped.Count, UnmappedXmlMemberIds = members.Keys.Except(mapped).Order().ToArray(),
                MethodSignaturesChecked = methodChecks, ParameterNamesChecked = parameterChecks, NonVoidReturnsChecked = returnChecks,
                VoidMethodsWithoutRequiredReturns = voidMethods, ConstructorsWithoutRequiredReturns = constructorChecks, PropertyValuesChecked = propertyChecks,
                UndocumentedPublicSymbols = undocumentedPublic.Order().ToArray(), DocumentationProblems = problems,
                SynthesizedPropertiesDocumentedBySummaryWithoutValueTag = generatedPropertySummaries.Order().ToArray(),
                PublicParameterlessConstructorsWithTypeSummaryOnly = parameterlessConstructors.Order().ToArray(),
                DeclaredExceptionTags = members.Values.Sum(e => e.Elements("exception").Count()),
                ExceptionScope = "Counts and Chinese text of declared XML exception tags only; propagation through called methods and full semantic exception coverage are not inferred.",
                DocumentationScope = "Compiled XML and reflection-mapped declarations, including public API. Compiler-generated members without XML are excluded. Positional record properties carry the compiler-generated Chinese summary from constructor param documentation; a separate value tag is not generated. Public parameterless constructors with only a type summary are listed separately because reflection does not distinguish implicit default constructors from explicit empty constructors. Unmapped XML IDs remain listed for review. No source-wide 100% claim."
            };
            File.WriteAllText(args[2], JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine(JsonSerializer.Serialize(new { report.IlOnly, report.XmlMembers, report.MappedXmlMembers, Problems = problems.Count, UndocumentedPublic = undocumentedPublic.Count }));
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    /// <summary>检查说明文字是否至少含一个汉字，不把这个结构检查解释为文档质量评分。</summary>
    /// <param name="text">XML 标签内的纯文本。</param>
    /// <returns>存在汉字时为 true。</returns>
    private static bool Chinese(string? text) => text is not null && Regex.IsMatch(text, "[\\u3400-\\u4dbf\\u4e00-\\u9fff]");
    /// <summary>识别编译器合成成员，避免把闭包或记录合成代码当成源码声明。</summary>
    /// <param name="member">待检查的反射成员。</param>
    /// <returns>成员由编译器生成时为 true。</returns>
    private static bool Generated(MemberInfo? member) => member is not null && (member.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false) || member.Name.Contains('<'));
    /// <summary>把 CLR 参数类型转换为编译 XML 使用的文档标识格式。</summary>
    /// <param name="type">参数或返回值类型。</param>
    /// <returns>支持普通类型、泛型、数组、指针和引用参数的文档类型名。</returns>
    private static string TypeId(Type type)
    {
        if (type.IsByRef) return TypeId(type.GetElementType()!) + "@";
        if (type.IsPointer) return TypeId(type.GetElementType()!) + "*";
        if (type.IsArray) return TypeId(type.GetElementType()!) + (type.GetArrayRank() == 1 ? "[]" : "[" + string.Join(',', Enumerable.Repeat("0:", type.GetArrayRank())) + "]");
        if (type.IsGenericParameter) return (type.DeclaringMethod is null ? "`" : "``") + type.GenericParameterPosition;
        if (type.IsGenericType)
        {
            Type[] arguments = type.GetGenericArguments(); int used = 0;
            return string.Join('.', type.GetGenericTypeDefinition().FullName!.Split('+').Select(part =>
            {
                int tick = part.IndexOf('`'); if (tick < 0) return part;
                int count = int.Parse(part[(tick + 1)..], System.Globalization.CultureInfo.InvariantCulture);
                string formatted = part[..tick] + "{" + string.Join(',', arguments.Skip(used).Take(count).Select(TypeId)) + "}"; used += count; return formatted;
            }));
        }
        return (type.FullName ?? type.ToString()).Replace('+', '.');
    }
    /// <summary>生成方法或构造函数的编译 XML 标识，转换运算符包括返回类型后缀。</summary>
    /// <param name="method">要匹配 XML 的方法或构造函数。</param>
    /// <returns>方法文档标识。</returns>
    private static string MethodId(MethodBase method)
    {
        string name = method is ConstructorInfo ? method.IsStatic ? "#cctor" : "#ctor" : method.Name.Replace('.', '#');
        if (method.IsGenericMethod) name += "``" + method.GetGenericArguments().Length;
        string id = "M:" + method.DeclaringType!.FullName!.Replace('+', '.') + "." + name;
        var parameters = method.GetParameters();
        if (parameters.Length > 0) id += "(" + string.Join(',', parameters.Select(p => TypeId(p.ParameterType))) + ")";
        if (method is MethodInfo conversion && method.Name is "op_Implicit" or "op_Explicit") id += "~" + TypeId(conversion.ReturnType);
        return id;
    }
}
