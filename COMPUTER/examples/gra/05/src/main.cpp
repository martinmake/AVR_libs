#include <gra/renderer.h>

#define WINDOW_WIDTH  640.0
#define WINDOW_HEIGHT 420.0

#define ZOOM 1

#define SPEED 10

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Window window(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	VertexArray vertex_array;
	{
		Buffer::Vertex vertex_buffer;
		{
			float positions[2 * 6] =
			{
				 10,  10,
				100,  10,
				 10, 100,
				100, 100,
				200, 200,
				300, 300,
			};

			vertex_buffer.data(positions, 2 * 6 * sizeof(float));
		}
		vertex_array.vertex_buffer(vertex_buffer);

		Buffer::Vertex::Layout vertex_buffer_layout;
		vertex_buffer_layout.push<float>(2);
		vertex_array.layout(vertex_buffer_layout);
	}

	glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(-100.0, 0.0, 0.0)); // MOVE OBJECTS TO THE LEFT
	glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(+100.0, 0.0, 0.0)); // MOVE CAMERA  TO THE RIGHT
	glm::mat4 projection = glm::ortho<float>(0.0, window.width() / ZOOM, 0.0, window.height() / ZOOM);
	glm::mat4 mvp = model * view * projection;
	Program program("res/shaders/basic");
	{
		program.set_uniform("u_mvp", mvp);
		program.set_uniform("u_color", 0.7, 0.1, 0.5, 1.0);
		program.set_uniform("u_point_size", (float) 10.0);
	}
	window.on_key([&](Event::Window::Key event)
	{
		float x_velocity = 0.0;
		float y_velocity = 0.0;

		switch (event.key())
		{
			case 'H': x_velocity -= SPEED; break;
			case 'L': x_velocity += SPEED; break;
		}
		switch (event.key())
		{
			case 'J': y_velocity -= SPEED; break;
			case 'K': y_velocity += SPEED; break;
		}

		mvp = glm::translate(mvp, glm::vec3(x_velocity, y_velocity, 0.0));
		program.set_uniform("u_mvp", mvp);
	});

	Renderer renderer;
	while (!window.should_close()) renderer.render(window, [&]()
	{
		renderer.draw(DrawMode::POINTS, program, vertex_array);
	});

	return 0;
}
