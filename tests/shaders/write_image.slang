// Writes x + y * width into an R16_UINT (or any uint) storage image bound by push descriptor.
struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [format("unknown")] RWTexture2D<uint> image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < pc.width && id.y < pc.height)
    {
        image[id.xy] = id.x + id.y * pc.width;
    }
}
