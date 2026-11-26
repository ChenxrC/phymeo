#version 450
#extension GL_GOOGLE_include_directive : require
#include "floor_light.glsl"
layout(location=0) in vec3 position;
layout(location=2) in mat4 objectToWorld;
void main()
{
    gl_Position = lightClip((objectToWorld * vec4(position, 1.0)).xyz);
}
