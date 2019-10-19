#ifndef _UTIL_TTY_H_
#define _UTIL_TTY_H_

#define ESC "\033"

enum TTY_CODE : uint8_t
{
	RESET                        =   0,
	NORMAL                       =   0,
	BOLD                         =   1,
	FAINT                        =   2,
	ITALIC                       =   3,
	UNDERLINE                    =   4,
	SLOW_BLINK                   =   5,
	RAPID_BLINK                  =   6,
	REVERSE                      =   7,
	CONCEAL                      =   8,
	CROSSEDOUT                   =   9,
	FRAKTUR                      =  20,
	BOLD_OFF                     =  21,
	DOUBLE_UNDERLINE             =  21,
	NORMAL_COLOR                 =  22,
	NORMAL_INTENSITY             =  22,
	ITALIC_AND_FRAKTUR_OFF       =  23,
	UNDERLINE_OFF                =  24,
	BLINK_OFF                    =  25,
	INVERSE_OFF                  =  27,
	REVEAL                       =  28,
	CROSSEDOUT_OFF               =  29,
	FRAMED                       =  51,
	ENCIRCLED                    =  52,
	OVERLINED                    =  53,
	FRAMED_OFF                   =  54,
	ENCIRCLED_OFF                =  54,
	OVERLINED_OFF                =  55,
	IDEOGRAM_UNDERLINE           =  60,
	IDEOGRAM_DOUBLE_UNDERLINE    =  61,
	IDEOGRAM_OVERLINE            =  62,
	IDEOGRAM_DOUBLE_OVERLINE     =  63,
	IDEOGRAM_STRESS_MARKING      =  64,
	IDEOGRAM_ATTRIBUTES_OFF      =  65,

	PRIMARY_FONT                 =  10,
	DEFAULT_FONT                 =  10,
	ALTERNATE_FONT1              =  11,
	ALTERNATE_FONT2              =  12,
	ALTERNATE_FONT3              =  13,
	ALTERNATE_FONT4              =  14,
	ALTERNATE_FONT5              =  15,
	ALTERNATE_FONT6              =  16,
	ALTERNATE_FONT7              =  17,
	ALTERNATE_FONT8              =  18,
	ALTERNATE_FONT9              =  19,

	COLOR_RGB                    =   2, // 3 * 8bit
	COLOR_NUMBER                 =   5, // 8bit
	/* 8bit COLOR LAYOUT
		0X00-0X07:  STANDARD COLORS (SAME AS THE 4-BIT COLOURS)
		0X08-0X0F:  HIGH INTENSITY COLORS
		0X10-0XE7:  6 × 6 × 6 CUBE (216 COLORS): 16 + 36 × R + 6 × G + B (0 ≤ R, G, B ≤ 5)
		0XE8-0XFF:  GRAYSCALE FROM BLACK TO WHITE IN 24 STEPS
	*/

	FG_BLACK                     =  30,
	FG_RED                       =  31,
	FG_GREEN                     =  32,
	FG_YELLOW                    =  33,
	FG_BLUE                      =  34,
	FG_MAGENTA                   =  35,
	FG_CYAN                      =  36,
	FG_WHITE                     =  37,
	FG_BRIGHT_BLACK              =  90,
	FG_BRIGHT_RED                =  91,
	FG_BRIGHT_GREEN              =  92,
	FG_BRIGHT_YELLOW             =  93,
	FG_BRIGHT_BLUE               =  94,
	FG_BRIGHT_MAGENTA            =  95,
	FG_BRIGHT_CYAN               =  96,
	FG_BRIGHT_WHITE              =  97,
	FG_SET_COLOR                 =  38, // FOLLOW WITH "COLOR_NUMBER, n" or "COLOR_RGB, r, g, b"
	FG_DEFAULT                   =  39,

	BG_BLACK                     =  40,
	BG_RED                       =  41,
	BG_GREEN                     =  42,
	BG_YELLOW                    =  43,
	BG_BLUE                      =  44,
	BG_MAGENTA                   =  45,
	BG_CYAN                      =  46,
	BG_WHITE                     =  47,
	BG_BRIGHT_BLACK              = 100,
	BG_BRIGHT_RED                = 101,
	BG_BRIGHT_GREEN              = 102,
	BG_BRIGHT_YELLOW             = 103,
	BG_BRIGHT_BLUE               = 104,
	BG_BRIGHT_MAGENTA            = 105,
	BG_BRIGHT_CYAN               = 106,
	BG_BRIGHT_WHITE              = 107,
	BG_SET_COLOR                 =  48, // FOLLOW WITH "COLOR_NUMBER, n" or "COLOR_RGB, r, g, b"
	BG_DEFAULT                   =  49,
};

inline void _fstty(FILE* stream)
{
	putc('m', stream);
}
template <typename V>
void _fstty(FILE* stream, V value)
{
	fprintf(stream, "%u", static_cast<uint8_t>(value));
	_fstty(stream);
}
template <typename V, typename ... A>
void _fstty(FILE* stream, V value, A ... args)
{
	fprintf(stream, "%u;", static_cast<uint8_t>(value));
	_fstty(stream, args ...);
}
template <typename ... A>
void fstty(FILE* stream, A ... args)
{
	fprintf(stream, "\033[");
	_fstty(stream, args ...);
}

#define stty(...) fstty(stdout, __VA_ARGS__)

#endif
