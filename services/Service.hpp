#pragma once

class VulkanApp;

class Service {
public:
    virtual ~Service() = default;
    virtual void init(VulkanApp* app) = 0;
    virtual void cleanup() = 0;
};
