#include "gpl/canvas.h"

namespace Gpl
{
	Gra::Renderer Canvas::s_renderer;

	Canvas::Canvas(void)
	{
	}
	Canvas::~Canvas(void)
	{
	}

	void Canvas::render(void)
	{
		static glm::mat4 mvp = glm::ortho<float>(0, s_renderer.width(), 0, s_renderer.height());

		s_renderer.start_frame();

		for (std::shared_ptr<Primitive::Base>& primitive : m_primitives)
			primitive->draw(s_renderer, mvp);

		s_renderer.end_frame();
	}

	Canvas& Canvas::operator<<(Primitive::Base* primitive)
	{
		m_primitives.emplace_back(primitive);
		return *this;
	}
}
