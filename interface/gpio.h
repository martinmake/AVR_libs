#ifndef _INTERFACE_GPIO_H_
#define _INTERFACE_GPIO_H_

class IGpio
{
	public: // CONSTRUCTORS
		IGpio(void) { }
	public: // DESTRUCTORS
		virtual ~IGpio(void) { }

	public: // METHODS
		virtual void set  (void) = 0;
		virtual void clear(void) = 0;
		virtual void toggle(void) = 0;
		virtual bool is_high(void) const = 0;
		virtual bool is_low (void) const = 0;

		virtual void make_input (void) = 0;
		virtual void make_output(void) = 0;
		virtual bool is_input (void) const = 0;
		virtual bool is_output(void) const = 0;

		virtual void pull_up   (void) = 0;
		virtual void disconnect(void) = 0;
		virtual bool is_pulled_up   (void) const = 0;
		virtual bool is_disconnected(void) const = 0;

	public: // OPERATORS
		virtual IGpio& operator=(bool state) = 0;
		virtual IGpio& operator()(MODE mode) = 0;
		virtual operator bool(void) const = 0;
};

#endif
