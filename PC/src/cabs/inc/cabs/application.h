#ifndef _CABS_APPLICATION_H_
#define _CABS_APPLICATION_H_

#include <ncurses.h>
#include "cabs/screen.h"

class Application
{
	private:
		std::vector<std::shared_ptr<Screen>>           m_screens;
		std::vector<std::shared_ptr<Screen>>::iterator m_selected_screen;

	public:
		Application(void);
		~Application(void);

	public:
		void run(void);
		void setup_colors(void);
		void exit(int status);

	public:
		template <typename S>
		Application& operator<<(S& screen);
		Screen& operator[](int index);
};

template <typename S>
Application& Application::operator<<(S& screen)
{
	m_screens.push_back(std::make_shared<S>(screen));

	return *this;
}

extern Application application;

#endif
