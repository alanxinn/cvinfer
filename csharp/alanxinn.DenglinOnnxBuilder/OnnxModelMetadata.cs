using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace alanxinn.Denglin;

/// <summary>ONNX graph metadata needed by the Denglin builder.</summary>
public sealed class OnnxModelMetadata
{
    private OnnxModelMetadata(IReadOnlyList<OnnxTensorMetadata> inputs,
        IReadOnlyList<OnnxTensorMetadata> outputs)
    {
        Inputs = inputs;
        Outputs = outputs;
    }

    public IReadOnlyList<OnnxTensorMetadata> Inputs { get; }
    public IReadOnlyList<OnnxTensorMetadata> Outputs { get; }

    /// <summary>
    /// Reads only the ONNX protobuf graph metadata. It does not load ONNX Runtime
    /// and therefore remains usable before the Denglin Python environment exists.
    /// </summary>
    public static OnnxModelMetadata Read(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("ONNX path is required", nameof(path));
        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException("ONNX file was not found", fullPath);

        try
        {
            var bytes = File.ReadAllBytes(fullPath);
            var model = new ProtoReader(bytes);
            GraphMetadata? graph = null;
            while (model.TryReadField(out var field, out var wireType))
            {
                if (field == 7 && wireType == 2)
                    graph = ReadGraph(model.ReadMessage());
                else
                    model.Skip(wireType);
            }
            if (graph == null)
                throw new InvalidDataException("The ONNX file does not contain a graph.");
            return new OnnxModelMetadata(graph.Inputs, graph.Outputs);
        }
        catch (OnnxMetadataException error)
        {
            throw new InvalidDataException("Unable to read ONNX graph metadata: " + error.Message, error);
        }
        catch (EndOfStreamException error)
        {
            throw new InvalidDataException("Unable to read ONNX graph metadata: truncated protobuf.", error);
        }
    }

    private static GraphMetadata? ReadGraph(ProtoReader reader)
    {
        var inputs = new List<OnnxTensorMetadata>();
        var outputs = new List<OnnxTensorMetadata>();
        var initializerNames = new HashSet<string>(StringComparer.Ordinal);

        while (reader.TryReadField(out var field, out var wireType))
        {
            switch (field)
            {
                case 5 when wireType == 2: // GraphProto.initializer
                    var initializer = reader.ReadMessage();
                    var name = ReadInitializerName(initializer);
                    if (!string.IsNullOrWhiteSpace(name)) initializerNames.Add(name);
                    break;
                case 11 when wireType == 2: // GraphProto.input
                    inputs.Add(ReadValueInfo(reader.ReadMessage()));
                    break;
                case 12 when wireType == 2: // GraphProto.output
                    outputs.Add(ReadValueInfo(reader.ReadMessage()));
                    break;
                default:
                    reader.Skip(wireType);
                    break;
            }
        }

        // Initializers are graph inputs in protobuf, but are not runtime inputs.
        inputs = inputs.Where(value => !initializerNames.Contains(value.Name)).ToList();
        return new GraphMetadata(inputs, outputs);
    }

    private static string? ReadInitializerName(ProtoReader reader)
    {
        string? name = null;
        while (reader.TryReadField(out var field, out var wireType))
        {
            if (field == 8 && wireType == 2)
                name = reader.ReadString();
            else
                reader.Skip(wireType);
        }
        return name;
    }

    private static OnnxTensorMetadata ReadValueInfo(ProtoReader reader)
    {
        string? name = null;
        var dataType = 0;
        var shape = Array.Empty<long>();
        while (reader.TryReadField(out var field, out var wireType))
        {
            switch (field)
            {
                case 1 when wireType == 2:
                    name = reader.ReadString();
                    break;
                case 2 when wireType == 2:
                    ReadType(reader.ReadMessage(), ref dataType, ref shape);
                    break;
                default:
                    reader.Skip(wireType);
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(name))
            throw new OnnxMetadataException("A graph value has no tensor name.");
        return new OnnxTensorMetadata(name, dataType, shape);
    }

    private static void ReadType(ProtoReader reader, ref int dataType, ref long[] shape)
    {
        while (reader.TryReadField(out var field, out var wireType))
        {
            if (field != 1 || wireType != 2)
            {
                reader.Skip(wireType);
                continue;
            }

            var tensor = reader.ReadMessage();
            while (tensor.TryReadField(out var tensorField, out var tensorWireType))
            {
                switch (tensorField)
                {
                    case 1 when tensorWireType == 0:
                        dataType = checked((int)tensor.ReadVarint());
                        break;
                    case 2 when tensorWireType == 2:
                        shape = ReadShape(tensor.ReadMessage());
                        break;
                    default:
                        tensor.Skip(tensorWireType);
                        break;
                }
            }
        }
    }

    private static long[] ReadShape(ProtoReader reader)
    {
        var dimensions = new List<long>();
        while (reader.TryReadField(out var field, out var wireType))
        {
            if (field == 1 && wireType == 2)
            {
                dimensions.Add(ReadDimension(reader.ReadMessage()));
            }
            else
            {
                reader.Skip(wireType);
            }
        }
        return dimensions.ToArray();
    }

    private static long ReadDimension(ProtoReader reader)
    {
        long? value = null;
        while (reader.TryReadField(out var field, out var wireType))
        {
            switch (field)
            {
                case 1 when wireType == 0:
                    value = unchecked((long)reader.ReadVarint());
                    break;
                case 2 when wireType == 2:
                    // A symbolic dim_param is represented as -1 for the builder.
                    reader.ReadString();
                    value = -1;
                    break;
                default:
                    reader.Skip(wireType);
                    break;
            }
        }
        return value.GetValueOrDefault(-1);
    }

    private sealed class GraphMetadata
    {
        public GraphMetadata(IReadOnlyList<OnnxTensorMetadata> inputs,
            IReadOnlyList<OnnxTensorMetadata> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
        public IReadOnlyList<OnnxTensorMetadata> Inputs { get; }
        public IReadOnlyList<OnnxTensorMetadata> Outputs { get; }
    }

    private sealed class OnnxMetadataException : Exception
    {
        public OnnxMetadataException(string message) : base(message) { }
    }

    private sealed class ProtoReader
    {
        private readonly byte[] bytes;
        private int offset;

        public ProtoReader(byte[] bytes) { this.bytes = bytes ?? throw new ArgumentNullException(nameof(bytes)); }

        public bool TryReadField(out int field, out int wireType)
        {
            if (offset >= bytes.Length)
            {
                field = 0;
                wireType = 0;
                return false;
            }
            var tag = ReadVarint();
            field = checked((int)(tag >> 3));
            wireType = checked((int)(tag & 7));
            if (field <= 0) throw new OnnxMetadataException("Invalid protobuf field number.");
            return true;
        }

        public ulong ReadVarint()
        {
            ulong value = 0;
            for (var index = 0; index < 10; index++)
            {
                if (offset >= bytes.Length) throw new EndOfStreamException();
                var current = bytes[offset++];
                value |= (ulong)(current & 0x7f) << (7 * index);
                if ((current & 0x80) == 0) return value;
            }
            throw new OnnxMetadataException("Invalid protobuf varint.");
        }

        public ProtoReader ReadMessage()
        {
            var length = checked((int)ReadVarint());
            if (length < 0 || bytes.Length - offset < length) throw new EndOfStreamException();
            var message = new byte[length];
            Buffer.BlockCopy(bytes, offset, message, 0, length);
            offset += length;
            return new ProtoReader(message);
        }

        public string ReadString()
        {
            var length = checked((int)ReadVarint());
            if (length < 0 || bytes.Length - offset < length) throw new EndOfStreamException();
            var text = Encoding.UTF8.GetString(bytes, offset, length);
            offset += length;
            return text;
        }

        public void Skip(int wireType)
        {
            switch (wireType)
            {
                case 0:
                    ReadVarint();
                    return;
                case 1:
                    SkipBytes(8);
                    return;
                case 2:
                    SkipBytes(checked((int)ReadVarint()));
                    return;
                case 3:
                    while (TryReadField(out _, out var nestedWireType))
                    {
                        if (nestedWireType == 4) return;
                        Skip(nestedWireType);
                    }
                    throw new EndOfStreamException();
                case 4:
                    return;
                case 5:
                    SkipBytes(4);
                    return;
                default:
                    throw new OnnxMetadataException("Unsupported protobuf wire type " + wireType + ".");
            }
        }

        private void SkipBytes(int count)
        {
            if (count < 0 || bytes.Length - offset < count) throw new EndOfStreamException();
            offset += count;
        }
    }
}

/// <summary>Metadata for one ONNX tensor value.</summary>
public sealed class OnnxTensorMetadata
{
    internal OnnxTensorMetadata(string name, int dataType, IReadOnlyList<long> shape)
    {
        Name = name;
        DataType = dataType;
        Shape = shape.ToArray();
    }

    public string Name { get; }
    public int DataType { get; }
    public IReadOnlyList<long> Shape { get; }
    public bool IsDynamic => Shape.Count == 0 || Shape.Any(value => value <= 0);
    public string DataTypeName => DataType switch
    {
        1 => "float32",
        2 => "uint8",
        3 => "int8",
        4 => "uint16",
        5 => "int16",
        6 => "int32",
        7 => "int64",
        9 => "bool",
        10 => "float16",
        11 => "float64",
        12 => "uint32",
        13 => "uint64",
        16 => "bfloat16",
        _ => DataType == 0 ? "undefined" : "onnx(" + DataType.ToString() + ")"
    };

    public string ShapeText => Shape.Count == 0
        ? "?"
        : "[" + string.Join(",", Shape.Select(value => value > 0 ? value.ToString() : "?")) + "]";
}
