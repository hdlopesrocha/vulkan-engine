// Forward declare SkyWidget to avoid including widget headers here
class SkyWidget;

template<typename T>
struct PassUBO {
    Buffer buffer;
    T data;

    PassUBO() = default;
    PassUBO(VulkanApp* app, size_t size) {
        buffer = app->createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
};
