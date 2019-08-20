#ifndef _GRA_WINDOW_H_
#define _GRA_WINDOW_H_

#include <functional>

#include "gra/core.h"
#include "gra/events/window/all.h"

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
			      bool          vsync (void) const;

			Math::vec2<float> mouse_position(void) const;
		public: // SETTERS
			void vsync(bool new_vsync);

			void on_resize        (Event::Window::Resize       ::callback new_on_resize        );
			void on_close         (Event::Window::Close        ::callback new_on_close         );
			void on_key           (Event::Window::Key          ::callback new_on_key           );
			void on_key_typed     (Event::Window::KeyTyped     ::callback new_on_key_typed     );
			void on_mouse_button  (Event::Window::MouseButton  ::callback new_on_mouse_button  );
			void on_mouse_scrolled(Event::Window::MouseScrolled::callback new_on_mouse_scrolled);
			void on_mouse_moved   (Event::Window::MouseMoved   ::callback new_on_mouse_moved   );

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
			bool         m_vsync;
		private:
			Event::Window::Resize       ::callback m_on_resize;
			Event::Window::Close        ::callback m_on_close;
			Event::Window::Key          ::callback m_on_key;
			Event::Window::KeyTyped     ::callback m_on_key_typed;
			Event::Window::MouseButton  ::callback m_on_mouse_button;
			Event::Window::MouseScrolled::callback m_on_mouse_scrolled;
			Event::Window::MouseMoved   ::callback m_on_mouse_moved;
		public:
			static Window* s_sharing_window;

		DECLARATION_MANDATORY(Window)
	};

	// FUNCTIONS
	inline bool Window::should_close(void)            const { return glfwWindowShouldClose(m_window);      }
	inline void Window::clear(void)                   const { glCall(glClear(GL_COLOR_BUFFER_BIT));        }
	inline void Window::on_update(void)               const { glfwSwapBuffers(m_window); glfwPollEvents(); }
	inline void Window::detatch_current_context(void)       { glfwMakeContextCurrent(NULL);                }

	// GETTERS
	DEFINITION_DEFAULT_GETTER(Window, width,        unsigned int )
	DEFINITION_DEFAULT_GETTER(Window, height,       unsigned int )
	DEFINITION_DEFAULT_GETTER(Window, title,  const std::string &)
	DEFINITION_DEFAULT_GETTER(Window, vsync,        bool         )
	// SETTERS
	inline void Window::vsync(bool new_vsync) { m_vsync = new_vsync; glfwSwapInterval(m_vsync); }
	DEFINITION_DEFAULT_SETTER(Window, on_resize,         Event::Window::Resize       ::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_close,          Event::Window::Close        ::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_key,            Event::Window::Key          ::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_key_typed,      Event::Window::KeyTyped     ::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_mouse_button,   Event::Window::MouseButton  ::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_mouse_scrolled, Event::Window::MouseScrolled::callback)
	DEFINITION_DEFAULT_SETTER(Window, on_mouse_moved,    Event::Window::MouseMoved   ::callback)
}

#endif
