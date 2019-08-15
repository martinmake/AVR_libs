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

	Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	float positions[2 * 4] =
	{
			  0,            0,
		MODEL_WIDTH,            0,
		MODEL_WIDTH, MODEL_HEIGHT,
			  0, MODEL_HEIGHT
	};
	Buffer::Vertex vertex_buffer(positions, 2 * 4 * sizeof(float));

	VertexArray::Layout vertex_buffer_layout;
	vertex_buffer_layout.push<float>(2);

	VertexArray vertex_array(vertex_buffer, vertex_buffer_layout);

	std::vector<uint32_t> indices =
	{
		0, 1, 2,
		2, 3, 0
	};
	Buffer::Index<uint32_t> index_buffer(indices);

	Program program("res/shaders/basic");
	{
		glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(MODEL_TRANSLATION_X, MODEL_TRANSLATION_Y, 0.0));
		glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0,                 0.0,                 0.0));
		glm::mat4 projection = glm::ortho(0.0, renderer.width() / ZOOM, 0.0, renderer.height() / ZOOM);
		glm::mat4 mvp = projection * view * model;

		program.set_uniform("u_mvp", mvp);
		program.set_uniform("u_color", 0.7, 0.1, 0.5, 1.0);
	}

	while (!renderer.should_close())
	{
		renderer.start_frame();

		renderer.draw(vertex_array, index_buffer, program, DrawMode::TRIANGLES);

		renderer.end_frame();
	}

	return 0;
}
