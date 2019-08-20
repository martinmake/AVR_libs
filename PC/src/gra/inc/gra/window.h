#ifndef _GRA_WINDOW_H_
#define _GRA_WINDOW_H_

#include <sml/sml.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

namespace Gra
{
	class Window
	{
		public: // CONSTRUCTORS
			Window(void);
			Window(int initial_width, int initial_height, const std::string& initial_title, bool initial_share_resources = true, bool initial_is_visible = true);

		public: // GETTERS
			      unsigned int  width (void) const;
			      unsigned int  height(void) const;
			const std::string & title (void) const;

		public: // FUNCTIONS
			bool should_close(void) const;
			void make_current() const;
			static void detatch_current_context();
			void clear() const;
			void on_update() const;

		private:
			GLFWwindow*  m_window;
			unsigned int m_width;
			unsigned int m_height;
			std::string  m_title;
			bool         m_share_resources;
			bool         m_is_visible;
		public:
			static Window* sharing_window;

		DECLARATION_MANDATORY(Window)
	};

	inline bool Window::should_close(void)            const { return glfwWindowShouldClose(m_window);      }
	inline void Window::make_current(void)            const { glfwMakeContextCurrent(m_window);            }
	inline void Window::clear(void)                   const { glCall(glClear(GL_COLOR_BUFFER_BIT));        }
	inline void Window::on_update(void)               const { glfwSwapBuffers(m_window); glfwPollEvents(); }
	inline void Window::detatch_current_context(void)       { glfwMakeContextCurrent(NULL);                }

	// GETTERS
	DEFINITION_DEFAULT_GETTER(Window, width,        unsigned int )
	DEFINITION_DEFAULT_GETTER(Window, height,       unsigned int )
	DEFINITION_DEFAULT_GETTER(Window, title,  const std::string &)
}

#endif
