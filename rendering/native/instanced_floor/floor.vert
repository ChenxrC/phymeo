#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in mat4 objectToWorld;
layout(location=6) in vec4 color;
layout(push_constant) uniform Camera { mat4 viewProjection; } camera;
layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(location=2) out vec3 albedo;
layout(location=3) flat out vec3 towardCamera;
void main()
{
    vec4 world = objectToWorld * vec4(position, 1.0);
    gl_Position = camera.viewProjection * world;
    worldPosition = world.xyz;
    // Instances use positive scales along orthogonal local axes.
    mat3 axes = mat3(objectToWorld);
    vec3 squared = vec3(dot(axes[0], axes[0]), dot(axes[1], axes[1]), dot(axes[2], axes[2]));
    worldNormal = normalize(axes * (normal / max(squared, vec3(1e-14))));
    albedo = color.rgb;
    towardCamera = -normalize(vec3(camera.viewProjection[0][3], camera.viewProjection[1][3],
                                   camera.viewProjection[2][3]));
}
