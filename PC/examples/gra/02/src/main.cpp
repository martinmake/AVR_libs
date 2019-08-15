#include <gra/renderer.h>

#define WINDOW_WIDTH  640.0
#define WINDOW_HEIGHT 420.0

#define MODEL_WIDTH  100.0
#define MODEL_HEIGHT 100.0

#define OFFSET 100

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	float positions1[2 * 4] =
	{
			  0,            0,
		MODEL_WIDTH,            0,
		MODEL_WIDTH, MODEL_HEIGHT,
			  0, MODEL_HEIGHT
	};
	Buffer::Vertex vertex_buffer1(positions1, 2 * 4 * sizeof(float));

	float positions2[2 * 3] =
	{
			  0 + OFFSET,            0 + OFFSET,
		MODEL_WIDTH +     10,            0 + OFFSET,
		MODEL_WIDTH + OFFSET, MODEL_HEIGHT +     10,
	};
	Buffer::Vertex vertex_buffer2(positions2, 2 * 3 * sizeof(float));

	VertexArray::Layout vertex_buffer_layout;
	vertex_buffer_layout.push<float>(2);

	VertexArray vertex_array(vertex_buffer1, vertex_buffer_layout);

	std::vector<uint32_t> indices1 =
	{
		0, 1, 2,
		2, 3, 0
	};
	Buffer::Index<uint32_t> index_buffer1(indices1);

	std::vector<uint32_t> indices2 =
	{
		0, 1, 2,
	};
	Buffer::Index<uint32_t> index_buffer2(indices2);

	Program program("res/shaders/basic");
	{
		glm::mat4 model      = glm::mat4(1.0);
		glm::mat4 view       = glm::mat4(1.0);
		glm::mat4 projection = glm::ortho<float>(0.0, renderer.width(), 0.0, renderer.height());
		glm::mat4 mvp = projection * view * model;

		program.set_uniform("u_mvp", mvp);
		program.set_uniform("u_color", 0.7, 0.1, 0.5, 1.0);
	}

	vertex_array.vertex_buffer(vertex_buffer2);
	while (!renderer.should_close())
	{
		renderer.start_frame();

		renderer.draw(vertex_array, index_buffer2, program, DrawMode::TRIANGLES);

		renderer.end_frame();
	}

	return 0;
}
