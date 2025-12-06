#include "vulkan/vulkan.hpp"

class MyApp : public VulkanApp {
    public:

        void setup() {
            
        };

        void update(float deltaTime) {

        };

        void draw() {

        };

};

int main() {
    MyApp app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

