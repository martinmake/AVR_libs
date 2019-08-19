#include <vendor/imgui/imgui.h>
#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

#include "logging.h"

#include "gra/window.h"

namespace Gra
{
	Window::Window(void)
		: m_window(nullptr), m_width(0), m_height(0), m_title("")
	{
	}
	Window::Window(int initial_width, int initial_height, const std::string& initial_title, const Window& share_resources)
		: Window()
	{
		init(initial_width, initial_height, initial_title, share_resources);
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

	void Window::init(int new_width, int new_height, const std::string& new_title, const Window& share_resources)
	{
		m_width  = new_width;
		m_height = new_height;
		m_title  = new_title;

		m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, share_resources.m_window);
		make_current();
		TRACE("GLFW: WINDOW: CREATED: {0}", (void*) m_window);
		glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height)
		{
			(void) window;
			glViewport(0, 0, width, height);
		});
		detatch_current_context();
	}
}
