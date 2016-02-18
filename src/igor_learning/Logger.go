/* a super simple Logger package
 'object' is created by declaring it in main() for example: var log Logger

It needs a filename/fullpath
	log.file_name("test.log").

It has a few simple, writing methods, which also create the file if it does not exist
	log.warn( "text" ) --> This appends 1 line into the file with 3 elements: timestamp	WARN	text
	log.debug( string )
	log.info( string )

It also has two helper methods;
	log.clean()		empties the file
	log.delete()
*/

package Logger

import (
	"os"
	"time"
)

// create an object 'logger' and then use that object to open up files, print to them,
type Logger struct {
	file_name 	string
}

// error function - not really sure if needed
func check(e error) {
	if e != nil {
		panic(e)
	}
}

// WRITING METHODS
func (l Logger) warn(s string) {
	l.log(s, "WARN")
}

func (l Logger) info(s string) {
	l.log(s, "INFO")
}

func (l Logger) debug(s string) {
	l.log(s, "DEBUG")
}

// OTHER METHODS
func (l Logger) clean() {
	os.Remove(l.file_name)
	os.Create(l.file_name)
}

func (l Logger) delete() {
	os.Remove(l.file_name)
}

func (l Logger) log(s, typ string) {

	// check if file does not exist, and then create it if that is the case
	if _, err := os.Stat( l.file_name ); os.IsNotExist(err) {
		os.Create(l.file_name)
	}

	// open file ( which definitelly exists )
	f, err := os.OpenFile(l.file_name, os.O_APPEND| os.O_WRONLY, 0600)

	if err != nil {
		panic(err)
	}

	defer f.Close() // pretty cool thing. It automatically closes the file at the end of the scope. 

	t := time.Now().Format(time.UnixDate)

	if _, err = f.WriteString(t + "\t" + typ + "\t" + s + "\r\n"); err != nil {
		panic(err)
	}
}

