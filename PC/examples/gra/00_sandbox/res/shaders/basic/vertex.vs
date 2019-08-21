#version 330

layout(location = 0) in vec4 position;

uniform mat4 u_mvp;
uniform float u_point_size;

void main()
{
	gl_Position = position;
	gl_PointSize = 100;
}
