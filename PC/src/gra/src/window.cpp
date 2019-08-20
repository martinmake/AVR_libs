#include <vendor/imgui/imgui.h>
#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

#include "logging.h"

#include "gra/window.h"

namespace Gra
{
	Window* Window::s_sharing_window;

	Window::Window(void)
		: m_window(nullptr), m_width(0), m_height(0), m_title(""), m_share_resources(false), m_is_visible(false)
	{
	}
	Window::Window(int initial_width, int initial_height, const std::string& initial_title, bool initial_share_resources, bool initial_is_visible)
		: m_width(initial_width), m_height(initial_height), m_title(initial_title), m_share_resources(initial_share_resources), m_is_visible(initial_is_visible)
	{
		glfwWindowHint(GLFW_VISIBLE, m_is_visible ? GLFW_TRUE : GLFW_FALSE);
		m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, m_share_resources ? s_sharing_window->m_window : nullptr);
		TRACE("GLFW: WINDOW: CREATED: {0}", (void*) m_window);

		make_current();
		if (m_is_visible)
		{
			glfwSetWindowUserPointer(m_window, this);
			vsync(true);

			glfwSetWindowCloseCallback(m_window, [](GLFWwindow* window)
			{
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);

				if (this_window.m_on_close)
				{
					Event::Window::Close event;
					this_window.m_on_close(event);
				}
			});
			glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				(void) scancode;
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);

				if (this_window.m_on_key)
				{
					Event::Window::Key event
					(
						key,
						static_cast<Event::Window::Key::Action>(action),
						static_cast<Event::Window::Key::Mod   >(mods  )
					); this_window.m_on_key(event);
				}
			});
			glfwSetCharCallback(m_window, [](GLFWwindow* window, unsigned int keycode)
			{
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);

				if (this_window.m_on_key_typed)
				{
					Event::Window::KeyTyped event(keycode);
					this_window.m_on_key_typed(event);
				}
			});
			glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods)
			{
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);

				if (this_window.m_on_mouse_button)
				{
					Event::Window::MouseButton event
					(
						static_cast<Event::Window::MouseButton::Button>(button),
						static_cast<Event::Window::MouseButton::Action>(action),
						static_cast<Event::Window::MouseButton::Mod   >(mods  )
					); this_window.m_on_mouse_button(event);
				}
			});
			glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos)
			{
				Window& this_window = *(Window*)glfwGetWindowUserPointer(window);

				if (this_window.m_on_mouse_moved)
				{
					Event::Window::MouseMoved event(xpos, ypos);
					this_window.m_on_mouse_moved(event);
				}
			});
			glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset, double yoffset)
			{
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);

				if (this_window.m_on_mouse_scrolled)
				{
					Event::Window::MouseScrolled event(xoffset, yoffset);
					this_window.m_on_mouse_scrolled(event);
				}
			});
			glfwSetWindowSizeCallback(m_window, [](GLFWwindow* window, int width, int height)
			{
				Window& this_window = *(Window*) glfwGetWindowUserPointer(window);
				this_window.m_width  = width;
				this_window.m_height = height;

				if (this_window.m_on_resize)
				{
					Event::Window::Resize event(width, height);
					this_window.m_on_resize(event);
				}
			});
		}

		TRACE("WINDOW: CONSTRUCTED: {0}", (void*) this);
	}

	Window::~Window(void)
	{
		if (m_window)
		{
			glfwDestroyWindow(m_window);
			TRACE("GLFW: WINDOW: DESTROYED: {0}", (void*) m_window);
		}
		TRACE("WINDOW: DESTRUCTED: {0}", (void*) this);
	}
}
