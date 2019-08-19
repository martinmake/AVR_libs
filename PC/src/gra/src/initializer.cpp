#include "initializer.h"
#include "logging.h"

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include <vendor/imgui/imgui.h>
#include <vendor/imgui/impl_glfw.h>
#include <vendor/imgui/impl_opengl3.h>

namespace Gra
{
	static Initializer initializer;

	Initializer::Initializer(void)
	{
		#ifdef LOG
			logger = new Logger::Console("GRA");
		#endif

		glfwSetErrorCallback([](int error, const char* description)
		{
			ERROR("GLFW: ERROR: {0} ({1})", description, error)
		});

		assert(glfwInit() && "GLFW init");
		INFO("GLFW: INITIALIZED");

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* window = glfwCreateWindow(1, 1, "", NULL, NULL);
		glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
		glfwMakeContextCurrent(window);
		glfwSwapInterval(1);

		assert(glewInit() == GLEW_OK && "GLEW init");

		TRACE("GL: VERSION: {0}",   glGetString(GL_VERSION)                 );
		TRACE("GLSL: VERSION: {0}", glGetString(GL_SHADING_LANGUAGE_VERSION));

		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
		TRACE("GL: BLENDING ENABLED");

	//	IMGUI_CHECKVERSION();
	//	ImGui::CreateContext();
	//	ImGui::StyleColorsDark();
	//
	//	ImGui_ImplGlfw_InitForOpenGL(window, true);
	//	ImGui_ImplOpenGL3_Init("#version 330");

		glfwDestroyWindow(window);
		INFO("INITIALIZED");
	}

	Initializer::~Initializer(void)
	{
	//	ImGui_ImplOpenGL3_Shutdown();
	//	ImGui_ImplGlfw_Shutdown();
	//	ImGui::DestroyContext();

		glfwTerminate();
		INFO("GLFW: TERMINATED");

		INFO("UNINITIALIZED");
		delete logger;
	}
}
