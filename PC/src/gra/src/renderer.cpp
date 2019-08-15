#include <vendor/imgui/imgui.h>

#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

#include "gra/renderer.h"
#include "gra/glstd.h"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[GLFW_ERROR] %d: %s\n", error, description);
}

namespace Gra
{
	Renderer::Renderer(void)
	{
	}
	Renderer::Renderer(int initial_width, int initial_height, const std::string& title)
	{
		init(initial_width, initial_height, title);
	}

	Renderer::~Renderer(void)
	{
		if (!m_window) return;

		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	void Renderer::init(int initial_width, int initial_height, const std::string& title)
	{
		m_width  = initial_width;
		m_height = initial_height;

		glfwSetErrorCallback(glfw_error_callback);
		assert(glfwInit() && "GLFW init");

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		m_window = glfwCreateWindow(m_width, m_height, title.c_str(), NULL, NULL);
		glfwMakeContextCurrent(m_window);
		glfwSwapInterval(1);

		assert(glewInit() == GLEW_OK && "GLEW init");

		glCall(std::cout << "[GL   VERSION] " << glGetString(GL_VERSION)                  << std::endl);
		glCall(std::cout << "[GLSL VERSION] " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl);

		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(m_window, true);
		ImGui_ImplOpenGL3_Init("#version 330");
	}

	void Renderer::start_frame() const
	{
		glCall(glClear(GL_COLOR_BUFFER_BIT));
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void Renderer::end_frame() const
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
}
