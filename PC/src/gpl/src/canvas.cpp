#include "gpl/canvas.h"

#include "gpl/primitives/shapes/point.h"
namespace Gpl
{
	using namespace Gra;
	using namespace Gra::GraphicsObject;
	using namespace Gra::Math;

	Canvas::Canvas(int initial_width, int initial_height, const std::string initial_title)
		: m_primitives(vec2<unsigned int>(0, 0), vec2<unsigned int>(initial_width, initial_height))
	{
		m_window = Gra::Window(initial_width, initial_height, initial_title);
	}

	Canvas::~Canvas(void)
	{
	}

	void Canvas::animate(void)
	{
		static Gra::Renderer renderer;

		Primitive::Shape::Point p(vec2<unsigned int>(100, 100), vec4<float>(0.5, 0.0, 0.5, 1.0), 80);
		while (!m_window.should_close()) renderer.render(m_window, [&]()
		{
			glm::mat4 mvp = glm::ortho<float>(0, m_window.width(), 0, m_window.height());
			m_primitives.draw(m_window.resolution(), mvp);
		});
	}

	void Canvas::copy(const Canvas& other)
	{
		m_primitives = other.m_primitives;
		m_window     = other.m_window;
	}
	void Canvas::move(Canvas&& other)
	{
		m_primitives = std::move(other.m_primitives);
		m_window     = std::move(other.m_window);
	}
}
