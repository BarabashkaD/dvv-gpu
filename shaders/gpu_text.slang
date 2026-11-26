// Text from a glyph strip (dvv/gpu/glyph_strip.h), for shaders: `import dvv_gpu_text.gpu_text;`,
// then `dvv::gpu::decimalCoverage` or `using namespace dvv::gpu;`. The strip is a row of equal
// cells, one character each, the coverage in every channel (an RGBA8 storage image, as
// uploadGlyphStrip makes it). The strip cannot hold its cell count: the caller passes it.
module gpu_text;

namespace dvv::gpu
{

// The cell grid of a strip: every cell `advance` pixels wide and `height` rows high. Both are 0
// without a usable grid.
public struct GlyphCells
{
    public uint advance;
    public uint height;
};

// The grid of `strip` with `cellCount` cells. Both fields are 0 when `cellCount` is 0 or a cell
// would be 0 pixels wide.
public GlyphCells glyphCells(RWTexture2D<float4> strip, uint cellCount)
{
    GlyphCells cells;
    cells.advance = 0u;
    cells.height = 0u;
    if (cellCount == 0u)
    {
        return cells;
    }
    uint width;
    uint height;
    strip.GetDimensions(width, height);
    const uint advance = width / cellCount;
    if (advance == 0u)
    {
        return cells;
    }
    cells.advance = advance;
    cells.height = height;
    return cells;
}

// The number of decimal digits of `value`: 1 for 0.
public uint decimalDigits(uint value)
{
    uint digits = 1u;
    for (; value >= 10u; value = value / 10u)
    {
        digits = digits + 1u;
    }
    return digits;
}

// The digit of `value` at `position`, 0 being the most significant of its `digits` digits
// (decimalDigits(value)); 0 for a position past the last.
public uint decimalDigitAt(uint value, uint digits, uint position)
{
    if (position >= digits)
    {
        return 0u;
    }
    uint power = 1u;
    for (uint i = position + 1u; i < digits; i = i + 1u)
    {
        power = power * 10u;
    }
    return (value / power) % 10u;
}

// The coverage at pixel (gx, gy) of cell `cell`, 0..1: the strip at (cell * advance + gx, gy); 0
// outside [0, advance) x [0, height) and for a cell past the strip.
// The strip's width must be cellCount * advance.
public float cellCoverage(RWTexture2D<float4> strip, GlyphCells cells, uint cell, int gx, int gy)
{
    if (cells.advance == 0u || gx < 0 || gx >= int(cells.advance) || gy < 0 ||
        gy >= int(cells.height))
    {
        return 0.0;
    }
    uint width;
    uint height;
    strip.GetDimensions(width, height);
    if (cell >= width / cells.advance)
    {
        return 0.0;
    }
    return strip[uint2(cell * cells.advance + uint(gx), uint(gy))].x;
}

// The coverage at `offset` pixels from the first pixel of the text of `value` and glyph row `gy`;
// the digits are in cells 0 to 9. 0 outside the text.
public float decimalCoverage(RWTexture2D<float4> strip, GlyphCells cells, uint value, int offset,
                             int gy)
{
    if (cells.advance == 0u || offset < 0)
    {
        return 0.0;
    }
    const uint digits = decimalDigits(value);
    if (uint(offset) >= digits * cells.advance)
    {
        return 0.0;
    }
    const uint cell = uint(offset) / cells.advance;
    const uint digit = decimalDigitAt(value, digits, cell);
    return cellCoverage(strip, cells, digit, int(uint(offset) % cells.advance), gy);
}

}  // namespace dvv::gpu
