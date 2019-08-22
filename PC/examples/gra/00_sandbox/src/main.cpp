#include <gra/renderer.h>

#define WINDOW_WIDTH  640.0
#define WINDOW_HEIGHT 420.0

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Window window(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	const float positions[2][2] =
	{
		{ -0.5, -0.5 },
		{ +0.5, +0.5 },
       	};
	Buffer::Vertex vertex_buffer1(positions[0], 2 * sizeof(float));
	Buffer::Vertex vertex_buffer2(positions[1], 2 * sizeof(float));

	VertexArray vertex_array;
	{
		Buffer::Vertex::Layout vertex_buffer_layout;
		vertex_buffer_layout.push<float>(2);

		vertex_array.layout(vertex_buffer_layout);
	}
	Program program("res/shaders/basic");

	Renderer renderer;
	while (!window.should_close()) renderer.render(window, [&]()
	{
		vertex_array.vertex_buffer(vertex_buffer1);
		renderer.draw(DrawMode::POINTS, program, vertex_array);
		vertex_array.vertex_buffer(vertex_buffer2);
		renderer.draw(DrawMode::POINTS, program, vertex_array);
	});

	return 0;
}
