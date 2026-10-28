// Adds 1 to every uint32 of a buffer that the push constants pass by buffer device address.
struct PushConstants
{
    uint* data;  // buffer device address (a PhysicalStorageBuffer pointer)
    uint count;  // number of values
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < pc.count)
    {
        pc.data[id.x] = pc.data[id.x] + 1;
    }
}
