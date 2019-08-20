#include <vendor/imgui/imgui.h>
#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

#include "logging.h"

#include "gra/window.h"

namespace Gra
{
	Window* Window::sharing_window;

	Window::Window(void)
		: m_window(nullptr), m_width(0), m_height(0), m_title(""), m_share_resources(false), m_is_visible(false)
	{
	}
	Window::Window(int initial_width, int initial_height, const std::string& initial_title, bool initial_share_resources, bool initial_is_visible)
		: m_width(initial_width), m_height(initial_height), m_title(initial_title), m_share_resources(initial_share_resources), m_is_visible(initial_is_visible)
	{
		glfwWindowHint(GLFW_VISIBLE, m_is_visible ? GLFW_TRUE : GLFW_FALSE);
		m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, m_share_resources ? sharing_window->m_window : nullptr);
		TRACE("GLFW: WINDOW: CREATED: {0}", (void*) m_window);

		make_current();
		if (m_is_visible)
		{
			glfwSwapInterval(1);
			glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height)
			{
				(void) window;
				glViewport(0, 0, width, height);
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
