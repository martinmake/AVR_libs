#include <vendor/imgui/imgui.h>

#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

#include "renderer.h"
#include "glstd.h"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[GLFW_ERROR] %d: %s\n", error, description);
}

namespace Gra
{
	Renderer::Renderer(int initial_width, int initial_height, const std::string& title)
		: m_width(initial_width), m_height(initial_height)
	{
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

	Renderer::~Renderer(void)
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	void Renderer::draw(const VertexArray& vertex_array, const IndexBuffer& index_buffer, const Shader& shader, DrawMode mode) const
	{
		vertex_array.bind();
		index_buffer.bind();
		shader.bind();

		glCall(glDrawElements(DrawMode_to_GLenum(mode), index_buffer.count(), GL_UNSIGNED_INT, nullptr));
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
