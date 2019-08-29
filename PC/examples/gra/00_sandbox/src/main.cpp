#include <thread>

#include <gra/renderer.h>

#define MODEL_WIDTH  100.0
#define MODEL_HEIGHT 100.0

#define ZOOM 1.0

#define SHADER_DIRPATH "res/shaders/basic"

int main(void)
{
	using namespace Gra;
	using namespace Gra::GraphicsObject;

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
	{
		Renderer renderer;

		std::thread t1([&]()
		{
			Window window(640, 400, "WINDOW1");
			Program program(SHADER_DIRPATH);
			VertexArray vertex_array(vertex_buffer, vertex_buffer_layout);
			while (!window.should_close()) renderer.render(window, [&]()
			{
				glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
				glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
				glm::mat4 projection = glm::ortho(0.0, window.width() / ZOOM, 0.0, window.height() / ZOOM);
				glm::mat4 mvp = projection * view * model;

				program.set_uniform("u_mvp", mvp);
				program.set_uniform("u_color", 0.7, 0.1, 0.5, 1.0);
				renderer.draw(DrawMode::TRIANGLES, program, vertex_array, index_buffer);
			});
		});

		std::thread t2([&]()
		{
			Window window(400, 640, "WINDOW2");
			Program program(SHADER_DIRPATH);
			VertexArray vertex_array(vertex_buffer, vertex_buffer_layout);
			while (!window.should_close()) renderer.render(window, [&]()
			{
				glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(100.0, 100.0, 0.0));
				glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(  0.0,   0.0, 0.0));
				glm::mat4 projection = glm::ortho(0.0, window.width() / ZOOM, 0.0, window.height() / ZOOM);
				glm::mat4 mvp = projection * view * model;

				program.set_uniform("u_mvp", mvp);
				program.set_uniform("u_color", 0., 0.8, 0.9, 1.0);
				renderer.draw(DrawMode::TRIANGLES, program, vertex_array, index_buffer);
			});
		});

		t1.join();
		t2.join();
	}

	return 0;
}
