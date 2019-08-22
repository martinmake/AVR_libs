#define CREATE_DEFAULT_LOGGER
#include <logger/logger.h>

#include <gra/renderer.h>

#define WINDOW_WIDTH  ((float) 1024)
#define WINDOW_HEIGHT ((float) 1024)

#define BLOCK_COUNT_X ((float)  16)
#define BLOCK_COUNT_Y ((float) 512)

#define BLOCK_RESOLUTION_X ((float) WINDOW_WIDTH  / BLOCK_COUNT_X)
#define BLOCK_RESOLUTION_Y ((float) WINDOW_HEIGHT / BLOCK_COUNT_Y)

#define STARTING_TIME_SPEED ((float) 10)

#define SLOW_SCROLL_SPEED 1
#define FAST_SCROLL_SPEED 10

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	float time_velocity = STARTING_TIME_SPEED;
	Window window(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");
	window.on_mouse_scrolled([&](auto& event)
	{
		float acceleration = window.key(Window::SpecialKey::LEFT_SHIFT) ? FAST_SCROLL_SPEED : SLOW_SCROLL_SPEED;
		int8_t direction = -1 * event.yoffset();
		time_velocity += direction * acceleration;
	});

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

		for (uint16_t y = 0; y < BLOCK_COUNT_Y; y++)
		for (uint16_t x = 0; x < BLOCK_COUNT_X; x++)
		{
			glCall(glViewport(x * BLOCK_RESOLUTION_X, y * BLOCK_RESOLUTION_Y, BLOCK_RESOLUTION_X, BLOCK_RESOLUTION_Y));
			program.set_uniform("u_time", time + x + y * BLOCK_COUNT_X);
			renderer.draw(DrawMode::TRIANGLES, program, vertex_array, index_buffer);
		}

		time += time_velocity / 60.0;
	});

	return 0;
}
