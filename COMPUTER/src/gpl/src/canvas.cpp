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
		: Window(initial_width, initial_height, initial_title), m_primitives(vec2<unsigned int>(0, 0), vec2<unsigned int>(initial_width, initial_height))
	{
		on_mouse_moved([&](auto& event)
		{
			std::queue<std::pair<Primitive::Container&, Position>> queue;
			Position mouse_position(event.xpos(), event.ypos());

			queue.push({ m_primitives, m_primitives.position() });
			Primitive::Base* highest_primitive = nullptr;
			while(!queue.empty())
			{
				Primitive::Container& next = queue.front().first;
				for (std::unique_ptr<Primitive::Base>& primitive : next.primitives())
				{
					if (primitive->colides(mouse_position - (queue.front().second + next.position())))
					{
						if (primitive->is_container())
							queue.push({ *((Primitive::Container*) &*primitive), queue.front().second });
						else
							highest_primitive = &*primitive;
					}
				}
				queue.pop();
			}
			if (highest_primitive)
			{
				Event::Primitive::MouseOver mouse_over_event(highest_primitive);
				highest_primitive->on_mouse_over(mouse_over_event);
			}
		});
		on_mouse_button([&](auto& event)
		{
			std::queue<std::pair<Primitive::Container&, Position>> queue;
			Position mouse_position = this->mouse_position();

			queue.push({ m_primitives, m_primitives.position() });
			Primitive::Base* highest_primitive = nullptr;
			while(!queue.empty())
			{
				Primitive::Container& next = queue.front().first;
				for (std::unique_ptr<Primitive::Base>& primitive : next.primitives())
				{
					if (primitive->colides(mouse_position - (queue.front().second + next.position())))
					{
						if (primitive->is_container())
							queue.push({ *((Primitive::Container*) &*primitive), queue.front().second });
						else
							highest_primitive = &*primitive;
					}
				}
				queue.pop();
			}
			if (highest_primitive)
			{
				Event::Primitive::MouseButton mouse_button_event(event.button(),
				                                                 event.action(),
				                                                 event.mods  (),
										 highest_primitive);
				highest_primitive->on_mouse_button(mouse_button_event);
			}
		});
	}

	Canvas::~Canvas(void)
	{
	}

	void Canvas::animate(void)
	{
		static Gra::Renderer renderer;

		while (!should_close()) renderer.render(*this, [&]()
		{
			static glm::mat4 mvp = glm::ortho<float>(0, width(), 0, height());
			static std::queue<std::pair<Primitive::Container&, Data::Draw>> queue;

			queue.push({ m_primitives, { resolution(), mvp } });
			while(!queue.empty())
			{
				queue.front().first.draw(queue);
				queue.pop();
			}
		});
	}

	void Canvas::copy(const Canvas& other)
	{
		Gra::Window::copy(other);

		m_primitives = other.m_primitives;
	}
	void Canvas::move(Canvas&& other)
	{
		Gra::Window::move(std::move(other));

		m_primitives = std::move(other.m_primitives);
	}
}
