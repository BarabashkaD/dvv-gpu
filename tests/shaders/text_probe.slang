// Writes gpu_text's functions into an R32_UINT image (binding 0), one invocation per pixel; the
// glyph strip is binding 1. Coverage is stored as round(coverage * 255).
//   mode 0: decimalCoverage(value, x - base, y)
//   mode 1: cellCoverage(cell = value, x - base, y)
//   mode 2: row 0 decimalDigits(value), row 1 decimalDigitAt(value, digits, x), row 2 the
//           advance and row 3 the height of the grid, all per column x
import dvv_gpu_text.gpu_text;

using namespace dvv::gpu;

struct ProbePush
{
    uint mode;
    uint value;
    uint cellCount;
    int base;
    uint width;
    uint height;
};

[[vk::push_constant]] ConstantBuffer<ProbePush> pc;

[[vk::binding(0, 0)]] [format("r32ui")] RWTexture2D<uint> output;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> strip;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.width || id.y >= pc.height)
    {
        return;
    }
    const GlyphCells cells = glyphCells(strip, pc.cellCount);
    const int x = int(id.x) - pc.base;
    const int y = int(id.y);
    uint result = 0u;
    if (pc.mode == 0u)
    {
        result = uint(round(decimalCoverage(strip, cells, pc.value, x, y) * 255.0));
    }
    else if (pc.mode == 1u)
    {
        result = uint(round(cellCoverage(strip, cells, pc.value, x, y) * 255.0));
    }
    else if (id.y == 0u)
    {
        result = decimalDigits(pc.value);
    }
    else if (id.y == 1u)
    {
        result = decimalDigitAt(pc.value, decimalDigits(pc.value), id.x);
    }
    else if (id.y == 2u)
    {
        result = cells.advance;
    }
    else
    {
        result = cells.height;
    }
    output[id.xy] = result;
}
