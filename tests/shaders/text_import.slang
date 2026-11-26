// Imports the exported text module, as an application's shader does: the package test builds it
// against the installed package (dvv_import_shader_sources) and in-tree.
import dvv_gpu_text.gpu_text;

using namespace dvv::gpu;

[[vk::binding(0, 0)]] [format("r32ui")] RWTexture2D<uint> output;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> strip;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    const GlyphCells cells = glyphCells(strip, 10u);
    output[id.xy] = uint(decimalCoverage(strip, cells, 42u, int(id.x), int(id.y)) * 255.0);
}
