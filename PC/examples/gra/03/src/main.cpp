#include <thread>

#include <gra/renderer.h>

#define WINDOW_WIDTH  640.0
#define WINDOW_HEIGHT 420.0

#define MODEL_WIDTH  100.0
#define MODEL_HEIGHT 100.0

#define MODEL_TRANSLATION_X 100.0
#define MODEL_TRANSLATION_Y  50.0

#define ZOOM 1.0

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Renderer renderer;
	Window master_window = Window(1, 1, "");

	Buffer::Vertex vertex_buffer;
	{
		float positions[2 * 4] =
		{
				  0,            0,
			MODEL_WIDTH,            0,
			MODEL_WIDTH, MODEL_HEIGHT,
				  0, MODEL_HEIGHT
		};

		vertex_buffer.data(positions, 2 * 4 * sizeof(float));
	}

	Buffer::Vertex::Layout vertex_buffer_layout;
	vertex_buffer_layout.push<float>(2);

	Buffer::Index index_buffer;
	{
		std::vector<Buffer::Index::type> indices =
		{
			0, 1, 2,
			2, 3, 0
		};
		index_buffer.indices(indices);
	}
	Window::detatch_current_context();

	std::thread t1([&]()
	{
		Window rendering_window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, "RENDERING WINDOW", master_window);
		Program program("res/shaders/basic");
		VertexArray vertex_array(vertex_buffer, vertex_buffer_layout, rendering_window);
		while (!rendering_window.should_close()) renderer.render(rendering_window, [&]()
		{
			glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
			glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
			glm::mat4 projection = glm::ortho(0.0, WINDOW_WIDTH / ZOOM, 0.0, WINDOW_HEIGHT / ZOOM);
			glm::mat4 mvp = projection * view * model;
			program.set_uniform("u_mvp", mvp);
			program.set_uniform("u_color", 0.7, 0.1, 0.5, 1.0);
			renderer.draw(vertex_array, index_buffer, program, DrawMode::TRIANGLES);
		});
	});

	std::thread t2([&]()
	{
		Window rendering_window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, "RENDERING WINDOW", master_window);
		Program program("res/shaders/basic");
		VertexArray vertex_array(vertex_buffer, vertex_buffer_layout, rendering_window);
		while (!rendering_window.should_close()) renderer.render(rendering_window, [&]()
		{
			glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(MODEL_TRANSLATION_X, MODEL_TRANSLATION_Y, 0.0));
			glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0,                 0.0,                 0.0));
			glm::mat4 projection = glm::ortho(0.0, WINDOW_WIDTH / ZOOM, 0.0, WINDOW_HEIGHT / ZOOM);
			glm::mat4 mvp = projection * view * model;
			program.set_uniform("u_mvp", mvp);
			program.set_uniform("u_color", 0., 0.8, 0.9, 1.0);
			renderer.draw(vertex_array, index_buffer, program, DrawMode::TRIANGLES);
		});
	});

	t1.join();
	t2.join();

	return 0;
}
