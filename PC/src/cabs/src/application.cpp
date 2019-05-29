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

	m_status.attatch_to_window(stdscr);
	// m_console.attatch_to_window(stdscr);

	m_label_attr             = Cabs::Colors::RED_BLACK;
	m_border_attr            = Cabs::Colors::BLUE_BLACK;
	m_shadow_attr            = Cabs::Colors::YELLOW_BLACK;
	m_selected_attr          = Cabs::Colors::RED_BLACK | ACS_DIAMOND | A_BLINK;
	m_screen_background_attr = Cabs::Colors::BLACK_BLACK;
	m_widget_background_attr = Cabs::Colors::BLACK_BLACK;
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

	m_status.draw();

	while (1)
	{
		static int key;

		key = getch();

		if (m_status.mode() == Cabs::Mode::NORMAL)
		{
			// move when ^H, ^J, ^K, ^L are pressed
			switch (key)
			{
				case 's':
					Cabs::move = false;
					curs_set(0);
					attroff(A_STANDOUT);
					break;
			}
		}

		if (m_selected_screen != nullptr)
			m_selected_screen->handle_key(key);

		if (key == ESC)
			m_status.mode(Cabs::Mode::NORMAL);
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
