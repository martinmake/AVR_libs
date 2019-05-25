#include <dialog.h>

#include "cabs/application.h"

Application application;

Application::Application(void)
{
	initscr();
	cbreak();
	keypad(stdscr, true);
	noecho();
	curs_set(0);
	setup_colors();
	refresh();
}

Application::~Application(void)
{
	endwin();
}

void Application::exit(int status)
{
	std::exit(status);
}

void Application::run(void)
{
	for (std::shared_ptr<Screen> screen : m_screens)
		screen->draw();

	while (1)
	{
		static int key;

		key = getch();

		if (Cabs::move)
		{
			// move when ^H, ^J, ^K, ^L are pressed
			switch (key)
			{

			}
		}

		if (key == ESC)
			Cabs::move = true;

		m_screens.back()->handle_key(key);
	}
}

void Application::setup_colors(void)
{
	start_color();

	int colors[] =
	{
		COLOR_BLACK, COLOR_RED,
		COLOR_GREEN, COLOR_YELLOW,
		COLOR_BLUE,  COLOR_MAGENTA,
		COLOR_CYAN,  COLOR_WHITE
	};

	int avaible_colors = sizeof(colors) / sizeof(int);
	short pair = 1;
	for (int bg = 0; bg < avaible_colors; bg++)
	{
		for (int fg = 0; fg < avaible_colors; fg++)
			init_pair(pair++, colors[fg], colors[bg]);
	}
}

Screen& Application::operator[](int index)
{
	return *m_screens[index];
}
