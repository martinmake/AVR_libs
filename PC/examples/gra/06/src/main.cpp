#include <gra/renderer.h>

#define WINDOW_WIDTH  ((float) 1024)
#define WINDOW_HEIGHT ((float) 1024)

#define BLOCK_COUNT_X ((float) 32           )
#define BLOCK_COUNT_Y ((float) BLOCK_COUNT_X)

#define BLOCK_RESOLUTION_X ((float) WINDOW_WIDTH  / BLOCK_COUNT_X)
#define BLOCK_RESOLUTION_Y ((float) WINDOW_HEIGHT / BLOCK_COUNT_Y)

#define TIME_SPEED 1

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Window window(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	VertexArray vertex_array;
	{
		Buffer::Vertex vertex_buffer;
		{
			float positions[4 * 2] =
			{
				-1.0,  -1.0,
				+1.0,  -1.0,
				+1.0,  +1.0,
				-1.0,  +1.0,
			};

			vertex_buffer.data(positions, 4 * 2 * sizeof(float));
		}
		vertex_array.vertex_buffer(vertex_buffer);

		Buffer::Vertex::Layout vertex_buffer_layout;
		vertex_buffer_layout.push<float>(2);
		vertex_array.layout(vertex_buffer_layout);
	}

	Buffer::Index index_buffer;
	{
		std::vector<Buffer::Index::type> indices =
		{
			0, 1, 2,
			2, 3, 0,
		};
		index_buffer.indices(indices);
	}

	Program program("res/shaders/basic");
	program.set_uniform("u_resolution", WINDOW_WIDTH, WINDOW_HEIGHT);
	Renderer renderer;
	while (!window.should_close()) renderer.render(window, [&]()
	{
		static float time = 0.0;

		for (uint16_t y = 0; y < BLOCK_RESOLUTION_Y; y++)
		for (uint16_t x = 0; x < BLOCK_RESOLUTION_X; x++)
		{
			glCall(glViewport(x * BLOCK_RESOLUTION_X, y * BLOCK_RESOLUTION_Y, BLOCK_RESOLUTION_X, BLOCK_RESOLUTION_Y));
			program.set_uniform("u_time", time + x + y * BLOCK_COUNT_X);
			renderer.draw(DrawMode::TRIANGLES, program, vertex_array, index_buffer);
		}

		time += TIME_SPEED / 60.0;
	});

	return 0;
}
