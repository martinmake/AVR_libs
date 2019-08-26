#include <queue>

#include "gpl/canvas.h"
#include "gpl/primitives/base.h"
#include "gpl/data.h"

namespace Gpl
{
	using namespace Gra;
	using namespace Gra::GraphicsObject;
	using namespace Gra::Math;

	Canvas::Canvas(int initial_width, int initial_height, const std::string initial_title)
		: m_primitives(vec2<unsigned int>(0, 0), vec2<unsigned int>(initial_width, initial_height))
	{
		m_window = Gra::Window(initial_width, initial_height, initial_title);
		m_window.on_mouse_moved([&](auto& event)
		{
			std::queue<std::pair<Primitive::Container&, Position>> queue;
			Position mouse_position(event.xpos(), event.ypos());

			queue.push({ m_primitives, m_primitives.position() });
			Primitive::Base* highest_primitive;
			while(!queue.empty())
			{
				Primitive::Container& next = queue.front().first;
				if (next.colides(mouse_position - queue.front().second))
				{
					highest_primitive = &next;
					for (std::unique_ptr<Primitive::Base>& primitive : next.primitives())
					{
						if (primitive->colides(mouse_position - (queue.front().second + next.position())))
							highest_primitive = &*primitive;
					}
				}
				queue.pop();
			}
			std::cout << (void*) highest_primitive << std::endl;
		});
	}

	Canvas::~Canvas(void)
	{
	}

	void Canvas::animate(void)
	{
		static Gra::Renderer renderer;

		while (!m_window.should_close()) renderer.render(m_window, [&]()
		{
			static glm::mat4 mvp = glm::ortho<float>(0, m_window.width(), 0, m_window.height());
			static std::queue<std::pair<Primitive::Container&, Data::Draw>> queue;

			queue.push({ m_primitives, { m_window.resolution(), mvp } });
			while(!queue.empty())
			{
				queue.front().first.draw(queue);
				queue.pop();
			}
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
		m_window     = std::move(other.m_window    );
	}
}
