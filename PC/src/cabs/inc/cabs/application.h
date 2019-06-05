#ifndef _CABS_APPLICATION_H_
#define _CABS_APPLICATION_H_

#include <ncurses.h>

#include "cabs/cabs.h"
#include "cabs/screen.h"

class Application
{
	private:
		// Console m_console;

	public:
		std::vector<std::shared_ptr<Screen>> m_screens;
		            std::shared_ptr<Screen>  m_selected_screen;

	public:
		int m_label_attr;
		int m_border_attr;
		int m_shadow_attr;
		int m_selected_attr;
		int m_screen_background_attr;
		int m_widget_background_attr;
		int m_widget_gap;

	public:
		Application(void);
		~Application(void);

	public:
		void run(void);
		void exit(int status);

	private:
		void setup_colors(void);

	// OPERATORS
	public:
		template <typename S>
		Application& operator<<(S& screen);
		Screen& operator[](int index);

	// GETTERS
	public:
		int label_attr            (void) const;
		int border_attr           (void) const;
		int shadow_attr           (void) const;
		int selected_attr         (void) const;
		int screen_background_attr(void) const;
		int widget_background_attr(void) const;
		int widget_gap            (void) const;

	// SETTERS
	public:
		void label_attr            (int new_label_attr            );
		void border_attr           (int new_border_attr           );
		void shadow_attr           (int new_shadow_attr           );
		void selected_attr         (int new_selected_attr         );
		void screen_background_attr(int new_screen_background_attr);
		void widget_background_attr(int new_widget_background_attr);
		void widget_gap            (int new_widget_gap            );
};

template <typename S>
Application& Application::operator<<(S& screen)
{
	m_screens.push_back(std::make_shared<S>(screen));
	screen.widget_gap(m_widget_gap);

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
inline int Application::screen_background_attr(void) const
{
	return m_screen_background_attr;
}
inline int Application::widget_background_attr(void) const
{
	return m_widget_background_attr;
}
inline int Application::widget_gap(void) const
{
	return m_widget_gap;
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
inline void Application::screen_background_attr(int new_screen_background_attr)
{
	m_screen_background_attr = new_screen_background_attr;
}
inline void Application::widget_background_attr(int new_widget_background_attr)
{
	m_widget_background_attr = new_widget_background_attr;
}
inline void Application::widget_gap(int new_widget_gap)
{
	m_widget_gap = new_widget_gap;
}

extern Application application;

#endif
