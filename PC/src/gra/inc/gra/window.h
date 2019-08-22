#ifndef _GRA_WINDOW_H_
#define _GRA_WINDOW_H_

#include <functional>

#include "gra/core.h"
#include "gra/events/window/all.h"

namespace Gra
{
	class Window
	{
		public: // TYPES
			enum class SpecialKey;

		public: // CONSTRUCTORS
			Window(void);
			Window(int initial_width, int initial_height, const std::string& initial_title, bool initial_share_resources = true, bool initial_is_visible = true);

		public: // GETTERS
			                 unsigned int   width     (void) const;
			                 unsigned int   height    (void) const;
			      Math::vec2<unsigned int>  resolution(void) const;
			const std::string             & title     (void) const;
			      bool                      vsync     (void) const;

			Math::vec2<float> mouse_position(void   ) const;
			bool              key           (SpecialKey key) const;
			bool              key           (char       key) const;
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

		public: // TYPES
			enum class SpecialKey
			{
				UNKNOWN       = -1,
				WORLD_1       = 161, /* non-US #1 */
				WORLD_2       = 162, /* non-US #2 */
				ESCAPE        = 256,
				BACKSPACE     = 259,
				INSERT        = 260,
				DELETE        = 261,
				RIGHT         = 262,
				LEFT          = 263,
				DOWN          = 264,
				UP            = 265,
				PAGE_UP       = 266,
				PAGE_DOWN     = 267,
				HOME          = 268,
				END           = 269,
				CAPS_LOCK     = 280,
				SCROLL_LOCK   = 281,
				NUM_LOCK      = 282,
				PRINT_SCREEN  = 283,
				PAUSE         = 284,
				F1            = 290,
				F2            = 291,
				F3            = 292,
				F4            = 293,
				F5            = 294,
				F6            = 295,
				F7            = 296,
				F8            = 297,
				F9            = 298,
				F10           = 299,
				F11           = 300,
				F12           = 301,
				F13           = 302,
				F14           = 303,
				F15           = 304,
				F16           = 305,
				F17           = 306,
				F18           = 307,
				F19           = 308,
				F20           = 309,
				F21           = 310,
				F22           = 311,
				F23           = 312,
				F24           = 313,
				F25           = 314,
				KP_0          = 320,
				KP_1          = 321,
				KP_2          = 322,
				KP_3          = 323,
				KP_4          = 324,
				KP_5          = 325,
				KP_6          = 326,
				KP_7          = 327,
				KP_8          = 328,
				KP_9          = 329,
				KP_DECIMAL    = 330,
				KP_DIVIDE     = 331,
				KP_MULTIPLY   = 332,
				KP_SUBTRACT   = 333,
				KP_ADD        = 334,
				KP_ENTER      = 335,
				KP_EQUAL      = 336,
				LEFT_SHIFT    = 340,
				LEFT_CONTROL  = 341,
				LEFT_ALT      = 342,
				LEFT_SUPER    = 343,
				RIGHT_SHIFT   = 344,
				RIGHT_CONTROL = 345,
				RIGHT_ALT     = 346,
				RIGHT_SUPER   = 347,
				MENU          = 348,
				LAST          = MENU,
			};

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
	inline Math::vec2<unsigned int> Window::resolution(void) const { return { m_width, m_height }; }
	inline bool Window::key(SpecialKey key) const { return glfwGetKey(m_window, (int) key) == GLFW_TRUE ? true : false; }
	inline bool Window::key(char       key) const { return glfwGetKey(m_window,       key) == GLFW_TRUE ? true : false; }
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
