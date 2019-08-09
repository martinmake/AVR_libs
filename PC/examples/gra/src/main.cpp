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

	Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	float positions[2 * 4] =
	{
			  0,            0,
		MODEL_WIDTH,            0,
		MODEL_WIDTH, MODEL_HEIGHT,
			  0, MODEL_HEIGHT
	};
	VertexBuffer vertex_buffer(positions, 2 * 4 * sizeof(float));

	VertexBufferLayout vertex_buffer_layout;
	vertex_buffer_layout.push<float>(2);

	VertexArray vertex_array;
	vertex_array.add_buffer(vertex_buffer, vertex_buffer_layout);

	unsigned int indices[2 * 3] =
	{
		0, 1, 2,
		2, 3, 0
	};
	IndexBuffer index_buffer(indices, 2 * 3);

	Shader shader("res/shaders/basic");
	{
		glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(MODEL_TRANSLATION_X, MODEL_TRANSLATION_Y, 0.0));
		glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0,                 0.0,                 0.0));
		glm::mat4 projection = glm::ortho(0.0, renderer.width() / ZOOM, 0.0, renderer.height() / ZOOM);
		glm::mat4 mvp = projection * view * model;

		shader.set_uniform_mat4f("u_mvp", mvp);
		shader.set_uniform_4f("u_color", 0.7, 0.1, 0.5, 1.0);
	}

	while (!renderer.should_close())
	{
		renderer.start_frame();

		renderer.draw(vertex_array, index_buffer, shader);

		renderer.end_frame();
	}

	return 0;
}
