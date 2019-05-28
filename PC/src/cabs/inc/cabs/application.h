#ifndef _CABS_APPLICATION_H_
#define _CABS_APPLICATION_H_

#include <ncurses.h>
#include "cabs/screen.h"

class Application
{
	private:
		std::vector<std::shared_ptr<Screen>> m_screens;
		            std::shared_ptr<Screen>  m_selected_screen;

	public:
		int m_label_attr      = Cabs::Colors::RED_BLACK;
		int m_border_attr     = Cabs::Colors::BLUE_CYAN;
		int m_shadow_attr     = Cabs::Colors::BLUE_BLUE;
		int m_selected_attr   = A_REVERSE;
		int m_background_attr = Cabs::Colors::BLACK_BLACK;

	public:
		Application(void);
		~Application(void);

	public:
		void run(void);
		void setup_colors(void);
		void exit(int status);

	// OPERATORS
	public:
		template <typename S>
		Application& operator<<(S& screen);
		Screen& operator[](int index);

	// GETTERS
	public:
		int label_attr     (void) const;
		int border_attr    (void) const;
		int shadow_attr    (void) const;
		int selected_attr  (void) const;
		int background_attr(void) const;

	// SETTERS
	public:
		void label_attr     (int new_label_attr     );
		void border_attr    (int new_border_attr    );
		void shadow_attr    (int new_shadow_attr    );
		void selected_attr  (int new_selected_attr  );
		void background_attr(int new_background_attr);
};

template <typename S>
Application& Application::operator<<(S& screen)
{
	m_screens.push_back(std::make_shared<S>(screen));

	if (m_selected_screen == nullptr)
		m_selected_screen = m_screens.front();

	return *this;
}

// GETTERS
inline int Application::label_attr(void) const
{
	return m_label_attr;
}
inline int Application::border_attr(void) const
{
	return m_border_attr;
}
inline int Application::shadow_attr(void) const
{
	return m_shadow_attr;
}
inline int Application::selected_attr(void) const
{
	return m_selected_attr;
}
inline int Application::background_attr(void) const
{
	return m_background_attr;
}

// SETTERS
inline void Application::label_attr(int new_label_attr)
{
	m_label_attr = new_label_attr;
}
inline void Application::border_attr(int new_border_attr)
{
	m_border_attr = new_border_attr;
}
inline void Application::shadow_attr(int new_shadow_attr)
{
	m_shadow_attr = new_shadow_attr;
}
inline void Application::selected_attr(int new_selected_attr)
{
	m_selected_attr = new_selected_attr;
}
inline void Application::background_attr(int new_background_attr)
{
	m_background_attr = new_background_attr;
}

extern Application application;

#endif
