#version 330

layout(location = 0) out vec4 color;

uniform vec4 u_color;

void main()
{
	color = vec4(0.2, 0.3, 0.7, 1.0);
}
