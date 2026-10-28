// Writes x + y * width into `first` and x + y * width + 1 into `second`: two uint storage images
// bound by push descriptor (bindings 0 and 1 of set 0).
struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [format("unknown")] RWTexture2D<uint> first;
[[vk::binding(1, 0)]] [format("unknown")] RWTexture2D<uint> second;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < pc.width && id.y < pc.height)
    {
        first[id.xy] = id.x + id.y * pc.width;
        second[id.xy] = id.x + id.y * pc.width + 1;
    }
}
