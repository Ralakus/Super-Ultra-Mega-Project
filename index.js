document.addEventListener("alpine:init", () => {
    Alpine.data("global", () => ({
        displaySuper: false,
        displayUltra: false,
        displayMega: false,
        displayTeam: false,
        displayHeaderFinished: false,
        displayMessage: false,

        init() {
            setTimeout(() => {
                this.displaySuper = true;
            }, 500);
            setTimeout(() => {
                this.displayUltra = true;
            }, 1000);
            setTimeout(() => {
                this.displayMega = true;
            }, 1500);
            setTimeout(() => {
                this.displayTeam = true;
            }, 2000);
            setTimeout(() => {
                this.displayHeaderFinished = true;
            }, 2500);
            setTimeout(() => {
                this.displayMessage = true;
            }, 3000);
        }
    }));
});