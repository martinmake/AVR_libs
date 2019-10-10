#include <list>

#include <gra/renderer.h>

#define WINDOW_WIDTH  640.0
#define WINDOW_HEIGHT 420.0

#define ZOOM 1

#define DEFAULT_POINT_SIZE ((float) 1.0)
#define SCROLLING_SPEED    ((float) 1.0)

#define PEN_COLOR        0.7, 0.1, 0.5, 1.0
#define BACKGROUND_COLOR 0.0, 0.0, 0.0, 1.0

/*
   ! DISCLAIMER !

   THIS IS A NAIVE APPROACH AND JUST A QUICK TEST OF THE EVENT SYSTEM
   I ALMOST FEEL BAD FOR HOW I WROTE IT
*/

int main(void)
{
	using namespace Gra;
	using namespace GraphicsObject;

	Window window(WINDOW_WIDTH, WINDOW_HEIGHT, "EXAMPLE WINDOW");

	std::vector<float> positions = { };
	VertexArray vertex_array;
	{
		Buffer::Vertex vertex_buffer(positions.data(), positions.size() * sizeof(float));
		vertex_array.vertex_buffer(vertex_buffer);

		Buffer::Vertex::Layout vertex_buffer_layout;
		vertex_buffer_layout.push<float>(2);
		vertex_array.layout(vertex_buffer_layout);
	}

	Program program("res/shaders/basic");
	{
		glm::mat4 model      = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
		glm::mat4 view       = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, 0.0));
		glm::mat4 projection = glm::ortho<float>(0.0, window.width() / ZOOM, 0.0, window.height() / ZOOM);
		glm::mat4 mvp = projection * view * model;

		program.set_uniform("u_mvp", mvp);
		program.set_uniform("u_point_size", DEFAULT_POINT_SIZE);
	}
	window.on_mouse_scrolled([&](Event::Window::MouseScrolled event)
	{
		static float point_size = DEFAULT_POINT_SIZE;

		point_size += event.yoffset() * SCROLLING_SPEED;
		program.set_uniform("u_point_size", point_size);
	});

	bool draw  = false;
	bool erase = false;
	std::list<bool> draw_or_erase;
	window.on_mouse_button([&](Event::Window::MouseButton event)
	{
		using Action = Event::Window::MouseButton::Action;
		using Button = Event::Window::MouseButton::Button;

		if (event.button() == Button::LEFT)
			draw = event.action() == Action::PRESS ? true : false;
		if (event.button() == Button::RIGHT)
			erase = event.action() == Action::PRESS ? true : false;

		if (draw || erase)
		{
			Math::vec2<float> pos = window.mouse_position();

			positions.emplace_back(pos.x);
			positions.emplace_back(window.height() - pos.y);
			vertex_array.vertex_buffer().data(positions.data(), positions.size() * sizeof(float));
			draw_or_erase.push_back(draw ? true : false);
		}
	});
	window.on_mouse_moved([&](Event::Window::MouseMoved event)
	{
		if (draw || erase)
		{
			positions.emplace_back(event.xpos());
			positions.emplace_back(window.height() - event.ypos());
			vertex_array.vertex_buffer().data(positions.data(), positions.size() * sizeof(float));
			draw_or_erase.push_back(draw ? true : false);
		}
	});

	Renderer renderer;
	while (!window.should_close()) renderer.render(window, [&]()
	{
		std::list<bool>::iterator it = draw_or_erase.begin();
		for (uint32_t i = 0; i < draw_or_erase.size(); i++)
		{
			if (*it) program.set_uniform("u_color", PEN_COLOR);
			else     program.set_uniform("u_color", BACKGROUND_COLOR);
			renderer.draw(DrawMode::POINTS, program, vertex_array, i, 1);
			it++;
		}
	});

	return 0;
}
