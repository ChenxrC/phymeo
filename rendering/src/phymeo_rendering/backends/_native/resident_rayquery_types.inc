// Only loaded/enabled for an explicit hardware-shadow context.
struct RayDispatch {
    PFN_vkCreateAccelerationStructureKHR create{};
    PFN_vkDestroyAccelerationStructureKHR destroy{};
    PFN_vkGetAccelerationStructureBuildSizesKHR sizes{};
    PFN_vkCmdBuildAccelerationStructuresKHR build{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR address{};
    PFN_vkGetBufferDeviceAddress buffer_address{};
};
