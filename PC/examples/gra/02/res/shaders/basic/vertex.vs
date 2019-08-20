#version 330

layout(location = 0) in vec4 position;

uniform mat4 u_mvp;
uniform float u_point_size;

void main()
{
	gl_Position = u_mvp * position;
	gl_PointSize = u_point_size;
}
